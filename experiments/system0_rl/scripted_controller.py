"""
Scripted IK controller for block stacking using sim ground truth + G1_29_ArmIK.

State machine: APPROACH -> PRE_GRASP -> GRASP -> LIFT -> TRANSPORT -> DESCEND -> RELEASE -> RETREAT
Uses pinocchio+casadi IK solver for correct arm trajectories.
All IK targets are in ROBOT BASE FRAME (not world frame).
"""

import sys
import os
import torch
import numpy as np
from enum import IntEnum
from scipy.spatial.transform import Rotation

# Add lerobot to path for G1_29_ArmIK
sys.path.insert(0, os.path.expanduser("~/unitree_IL_lerobot/unitree_lerobot/lerobot/src"))


class Phase(IntEnum):
    APPROACH = 0
    PRE_GRASP = 1
    GRASP = 2
    LIFT = 3
    TRANSPORT = 4
    DESCEND = 5
    RELEASE = 6
    RETREAT = 7
    DONE = 8


PHASE_HOLD = {
    Phase.PRE_GRASP: 30,
    Phase.GRASP: 50,
    Phase.RELEASE: 30,
    Phase.RETREAT: 40,
}

FINGERS_OPEN = 0.0
FINGERS_CLOSED = 0.8  # Set conservatively; will be updated from joint limits


class ScriptedController:
    """
    Scripted controller using G1_29_ArmIK for proper IK.
    All positions are transformed from world frame to robot base frame
    before passing to the IK solver.
    """

    def __init__(self, config, device="cpu", sim_arm_indices=None, sim_hand_indices=None, env_idx=0):
        self.config = config
        self.device = device
        self.stacking_order = config.stacking_order
        self.block_height = config.block_height
        self.env_idx = env_idx

        # Explicit joint index mapping from train.py
        self.sim_arm_indices = sim_arm_indices
        self.sim_hand_indices = sim_hand_indices

        self.current_block_idx = 0
        self.phase = Phase.APPROACH
        self.phase_step = 0
        self.hand = "right"

        # Stack target (world frame)
        self.stack_target_world = np.array([-4.19, -3.95, 0.822])

        # Robot base pose — will be read from sim on first step
        self._robot_pos_w = None
        self._world_to_robot_rot = None
        self._initialized_transform = False

        # Load IK solver
        try:
            # Direct file import to avoid lerobot dependency chain
            import importlib.util
            _kin_path = os.path.expanduser(
                "~/unitree_IL_lerobot/unitree_lerobot/lerobot/src/lerobot/robots/unitree_g1/g1_kinematics.py"
            )
            _spec = importlib.util.spec_from_file_location("g1_kin", _kin_path)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            G1_29_ArmIK = _mod.G1_29_ArmIK
            self.ik_solver = G1_29_ArmIK(unit_test=False)
            self._has_ik = True
            print("[ScriptedController] G1_29_ArmIK loaded successfully")
            # Build pinocchio↔sim joint index mapping (critical for correct IK)
            import pinocchio as pin
            self._pin = pin
            model = self.ik_solver.reduced_robot.model
            self._pin_joint_names = []
            for jid in range(1, model.njoints):
                self._pin_joint_names.append(model.names[jid])
            print(f"[IK] Pinocchio joint order ({len(self._pin_joint_names)}): {self._pin_joint_names}")

            # FK at zero pose
            data = model.createData()
            q_zero = np.zeros(model.nq)
            pin.forwardKinematics(model, data, q_zero)
            pin.updateFramePlacements(model, data)
            for frame_name in ["L_ee", "R_ee"]:
                fid = model.getFrameId(frame_name)
                print(f"[IK FK] {frame_name} at q=0: pos={data.oMf[fid].translation}")

            # Step 3: Print IK joint order for debugging
            if hasattr(self.ik_solver, '_arm_joint_names_pin'):
                print(f"[IK] Pin arm order: {self.ik_solver._arm_joint_names_pin}")
            if hasattr(self.ik_solver, '_arm_joint_names_g1'):
                print(f"[IK] G1 arm order: {self.ik_solver._arm_joint_names_g1}")
            if hasattr(self.ik_solver, '_arm_reorder_g1_to_pin'):
                print(f"[IK] Reorder G1->pin: {self.ik_solver._arm_reorder_g1_to_pin}")
        except Exception as e:
            print(f"[ScriptedController] WARNING: G1_29_ArmIK failed: {e}")
            self.ik_solver = None
            self._has_ik = False

        self._current_arm_q = np.zeros(14)

        # Wrist orientation: will be read from sim default pose on first step
        self._grasp_rot = None  # Set from actual wrist orientation in _init_transform

        self._body_idx_cache = {}
        self._finger_limits_read = False
        self._first_step_done = False

        # Fix 4: Bypass IK with hardcoded arm positions to test joint mapping
        self._bypass_ik = False  # Joint mapping verified, now testing IK
        self._test_arm_pos = np.array([
            0.0, 0.3, 0.0, -0.5, 0.0, 0.0, 0.0,   # left arm (neutral)
            0.0, -0.3, 0.0, -0.5, 0.0, 0.0, 0.0,   # right arm (reaching forward)
        ])

    def _build_pin_sim_mapping(self, env):
        """Build mapping between pinocchio joint order and sim joint indices."""
        if not self._has_ik or hasattr(self, '_pin_to_sim_idx'):
            return
        joint_names = env.scene["robot"].data.joint_names
        name_to_sim = {name: i for i, name in enumerate(joint_names)}
        self._pin_to_sim_idx = []
        for pin_name in self._pin_joint_names:
            sim_idx = name_to_sim.get(pin_name)
            if sim_idx is None:
                print(f"[IK] WARNING: pin joint '{pin_name}' not in sim!")
                self._pin_to_sim_idx.append(0)
            else:
                self._pin_to_sim_idx.append(sim_idx)
        print(f"[IK] pin→sim index mapping: {self._pin_to_sim_idx}")

    def _sim_to_pin_joints(self, env):
        """Read current sim joint positions in pinocchio order."""
        full_pos = env.scene["robot"].data.joint_pos[self.env_idx].cpu().numpy()
        return np.array([full_pos[i] for i in self._pin_to_sim_idx])

    def _pin_to_sim_joints(self, pin_q, env):
        """Write pinocchio-order solution back to sim joint array."""
        full_action = env.scene["robot"].data.default_joint_pos[self.env_idx].cpu().numpy().copy()
        for pin_idx, sim_idx in enumerate(self._pin_to_sim_idx):
            full_action[sim_idx] = pin_q[pin_idx]
        return full_action

    def _init_transform(self, env):
        """Read robot base pose from sim and build world->robot transform."""
        if self._initialized_transform:
            return

        try:
            self._robot_pos_w = env.scene["robot"].data.root_pos_w[self.env_idx].cpu().numpy()
            quat_wxyz = env.scene["robot"].data.root_quat_w[self.env_idx].cpu().numpy()
            # scipy uses (x,y,z,w) format
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            rot = Rotation.from_quat(quat_xyzw)
            # Inverse rotation for world->robot
            self._world_to_robot_rot = rot.inv().as_matrix()
            self._initialized_transform = True
            print(f"[ScriptedController] Robot base: pos={self._robot_pos_w}, "
                  f"quat(wxyz)={quat_wxyz}")

            # Palm-down wrist orientation in robot base frame.
            # Z-axis of wrist points down (-Z in robot frame) = palm facing table.
            # X-axis points forward, Y-axis points left.
            self._grasp_rot = np.array([
                [1,  0,  0],   # wrist X → robot forward
                [0,  0,  1],   # wrist Y → robot up (fingers spread sideways)
                [0, -1,  0],   # wrist Z → robot left (palm down)
            ], dtype=float)
            print(f"[ScriptedController] Grasp rotation: palm-down")
        except Exception as e:
            print(f"[ScriptedController] WARNING: Could not read robot pose: {e}")
            # Fallback: hardcoded from scene config
            self._robot_pos_w = np.array([-4.2, -3.7, 0.76])
            angle = np.pi / 2
            c, s = np.cos(angle), np.sin(angle)
            self._world_to_robot_rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            self._initialized_transform = True

    def _init_finger_limits(self, env):
        """Read actual finger joint limits from sim."""
        if self._finger_limits_read:
            return
        global FINGERS_CLOSED
        try:
            robot = env.scene["robot"]
            if self.sim_hand_indices is not None:
                for local_i, sim_i in enumerate(self.sim_hand_indices):
                    name = robot.data.joint_names[sim_i]
                    lo = robot.data.soft_joint_pos_limits[0, sim_i, 0].item()
                    hi = robot.data.soft_joint_pos_limits[0, sim_i, 1].item()
                    print(f"  {name}: [{lo:.3f}, {hi:.3f}]")
            else:
                for i, name in enumerate(robot.data.joint_names):
                    if "hand" in name:
                        lo = robot.data.soft_joint_pos_limits[0, i, 0].item()
                        hi = robot.data.soft_joint_pos_limits[0, i, 1].item()
                        print(f"  {name}: [{lo:.3f}, {hi:.3f}]")
            # Use 80% of upper limit as closed position
            # Find a representative hand joint
            for i, name in enumerate(robot.data.joint_names):
                if "hand_thumb_0" in name:
                    hi = robot.data.soft_joint_pos_limits[0, i, 1].item()
                    FINGERS_CLOSED = hi * 0.8
                    print(f"[ScriptedController] FINGERS_CLOSED set to {FINGERS_CLOSED:.3f}")
                    break
        except Exception as e:
            print(f"[ScriptedController] Could not read finger limits: {e}")
        self._finger_limits_read = True

    def _world_to_robot(self, pos_world):
        """Transform world-frame position to robot base frame."""
        return self._world_to_robot_rot @ (pos_world - self._robot_pos_w)

    def _get_block_pos(self, env, block_name):
        return env.scene[block_name].data.root_pos_w[self.env_idx].cpu().numpy()

    def _get_body_idx(self, env, body_name):
        if body_name not in self._body_idx_cache:
            body_names = env.scene["robot"].data.body_names
            for i, name in enumerate(body_names):
                if body_name in name:
                    self._body_idx_cache[body_name] = i
                    break
            else:
                self._body_idx_cache[body_name] = -1
        return self._body_idx_cache[body_name]

    def _get_ee_pos_world(self, env, hand="right"):
        body_name = f"{hand}_wrist_yaw"
        idx = self._get_body_idx(env, body_name)
        if idx >= 0:
            return env.scene["robot"].data.body_pos_w[self.env_idx, idx].cpu().numpy()
        return self._robot_pos_w + np.array([0.0, 0.3 if hand == "left" else -0.3, 0.3])

    def _get_arm_joints(self, env):
        """Get current arm joint positions using explicit sim indices."""
        if self.sim_arm_indices is not None:
            return env.scene["robot"].data.joint_pos[self.env_idx, self.sim_arm_indices].cpu().numpy()
        # Fallback: old heuristic (should not be needed)
        full_pos = env.scene["robot"].data.joint_pos[self.env_idx].cpu().numpy()
        if len(full_pos) >= 29:
            return full_pos[15:29]
        elif len(full_pos) >= 28:
            return full_pos[-28:-14]
        return np.zeros(14)

    def _get_contact_force(self, env, hand="right"):
        try:
            forces = env.scene["fingertip_contacts"].data.net_forces_w[self.env_idx].cpu().numpy()
            if hand == "right":
                return np.linalg.norm(forces[3:6], axis=-1).sum()
            else:
                return np.linalg.norm(forces[0:3], axis=-1).sum()
        except (KeyError, AttributeError):
            return 0.0

    def _select_hand(self, env, target_pos_world):
        left_ee = self._get_ee_pos_world(env, "left")
        right_ee = self._get_ee_pos_world(env, "right")
        return "left" if np.linalg.norm(left_ee - target_pos_world) < np.linalg.norm(right_ee - target_pos_world) else "right"

    def _make_se3(self, position_robot, rotation=None):
        T = np.eye(4)
        T[:3, 3] = position_robot
        T[:3, :3] = rotation if rotation is not None else self._grasp_rot
        return T

    def _solve_ik(self, target_pos_robot, hand, env=None):
        """Solve IK. target_pos_robot must be in robot base frame.

        Uses pinocchio joint order internally, converts back to G1 motor order
        via the IK solver's built-in reordering.
        """
        if not self._has_ik:
            return self._current_arm_q

        # Get current joints in pinocchio order
        if env is not None and hasattr(self, '_pin_to_sim_idx'):
            current_pin_q = self._sim_to_pin_joints(env)
        else:
            current_pin_q = self._current_arm_q

        # FK to get current EE poses (in pinocchio/robot frame)
        model = self.ik_solver.reduced_robot.model
        data = model.createData()
        self._pin.forwardKinematics(model, data, current_pin_q)
        self._pin.updateFramePlacements(model, data)

        L_ee_id = model.getFrameId("L_ee")
        R_ee_id = model.getFrameId("R_ee")
        current_L = data.oMf[L_ee_id].homogeneous.copy()
        current_R = data.oMf[R_ee_id].homogeneous.copy()

        reach_T = self._make_se3(target_pos_robot)

        if hand == "left":
            left_T, right_T = reach_T, current_R
        else:
            left_T, right_T = current_L, reach_T

        try:
            sol_q, _ = self.ik_solver.solve_ik(
                left_wrist=left_T,
                right_wrist=right_T,
                current_lr_arm_motor_q=current_pin_q,
            )
            sol_q = np.array(sol_q, dtype=float)
            if np.any(np.isnan(sol_q)) or np.any(np.isinf(sol_q)):
                print(f"[IK NaN] phase={self.phase.name} hand={hand}")
                return self._current_arm_q
            if self.phase_step < 3:
                print(f"[IK DEBUG] target_robot={target_pos_robot} hand={hand}")
                print(f"  current L_ee={current_L[:3,3]}, R_ee={current_R[:3,3]}")
                print(f"  max_change={np.abs(sol_q - current_pin_q).max():.4f}")
            # sol_q is in pinocchio order — reorder to G1 motor order (ARM_NAMES)
            sol_g1 = sol_q[self.ik_solver._arm_reorder_pin_to_g1]
            self._current_arm_q = sol_g1
            return self._current_arm_q
        except Exception as e:
            print(f"[IK FAIL] phase={self.phase.name} hand={hand} "
                  f"target={target_pos_robot} error={e}")
            return self._current_arm_q

    def _compute_finger_targets(self, close=False, hand="right", env=None, retract=False):
        """Compute per-joint finger close targets using actual joint limits.
        retract=True: partially close fingers to avoid pushing objects during approach.
        """
        targets = np.zeros(14)
        if not close and not retract:
            return targets

        # Step 4: Use per-joint limits when available
        scale = 0.3 if retract else 0.7  # retract = 30% closed, full = 70%
        if env is not None and self.sim_hand_indices is not None:
            robot = env.scene["robot"]
            for local_i, sim_i in enumerate(self.sim_hand_indices):
                lo = robot.data.soft_joint_pos_limits[0, sim_i, 0].item()
                hi = robot.data.soft_joint_pos_limits[0, sim_i, 1].item()
                if abs(hi) > abs(lo):
                    targets[local_i] = hi * scale
                else:
                    targets[local_i] = lo * scale
            # Zero out the hand we are NOT using
            if hand == "left":
                targets[7:] = 0.0
            else:
                targets[:7] = 0.0
        else:
            # Fallback to global constant
            value = FINGERS_CLOSED
            if hand == "left":
                targets[:7] = value
            else:
                targets[7:] = value
        return targets

    def _make_phase_intent(self):
        intent = torch.zeros(128, device=self.device)
        if self.phase.value < 9:
            intent[self.phase.value] = 1.0
        intent[9 + min(self.current_block_idx, 2)] = 1.0
        intent[12] = 1.0 if self.hand == "left" else 0.0
        intent[13] = 1.0 if self.hand == "right" else 0.0
        return intent

    def step(self, env):
        """One step. All IK targets converted to robot base frame."""
        # Lazy init
        self._init_transform(env)
        self._init_finger_limits(env)

        # Step 2: Debug print actual wrist quaternion on first step
        if not self._first_step_done:
            self._first_step_done = True
            for hand_name in ["left", "right"]:
                body_name = f"{hand_name}_wrist_yaw"
                idx = self._get_body_idx(env, body_name)
                if idx >= 0:
                    quat = env.scene["robot"].data.body_quat_w[self.env_idx, idx].cpu().numpy()
                    pos = env.scene["robot"].data.body_pos_w[0, idx].cpu().numpy()
                    print(f"[Wrist Debug] {hand_name}_wrist_yaw: pos={pos}, quat(wxyz)={quat}")

        # Build pin↔sim mapping on first step
        self._build_pin_sim_mapping(env)

        # Cache world-frame EE positions
        self._ee_left_world = self._get_ee_pos_world(env, "left")
        self._ee_right_world = self._get_ee_pos_world(env, "right")
        self._current_arm_q = self._get_arm_joints(env)

        if self.current_block_idx >= len(self.stacking_order):
            self.phase = Phase.DONE
            fingers = self._compute_finger_targets(close=False, hand=self.hand, env=env)
            return torch.tensor(np.concatenate([self._current_arm_q, fingers]),
                                device=self.device, dtype=torch.float32), self._make_phase_intent()

        block_name = self.stacking_order[self.current_block_idx]
        block_pos_world = self._get_block_pos(env, block_name)

        place_pos_world = self.stack_target_world.copy()
        place_pos_world[2] += self.current_block_idx * self.block_height + self.block_height

        if self.phase == Phase.APPROACH and self.phase_step == 0:
            self.hand = self._select_hand(env, block_pos_world)

        ee_pos_world = self._get_ee_pos_world(env, self.hand)
        dist_to_block = np.linalg.norm(ee_pos_world - block_pos_world)
        contact = self._get_contact_force(env, self.hand)

        close_fingers = False

        # Fix 4: Bypass IK — test joint mapping with hardcoded positions
        if self._bypass_ik:
            arm = self._test_arm_pos.copy()
            close_fingers = True  # test finger closing too
            if self.phase_step % 50 == 0:
                print(f"[BYPASS IK] step={self.phase_step}")
                print(f"  arm[:7]={arm[:7]}")
                print(f"  arm[7:]={arm[7:]}")
                fingers_test = self._compute_finger_targets(close=True, hand="right", env=env)
                print(f"  fingers[7:]={fingers_test[7:]}")  # right hand targets
                # Also print current sim hand positions
                if self.sim_hand_indices is not None:
                    cur = env.scene["robot"].data.joint_pos[0, self.sim_hand_indices].cpu().numpy()
                    print(f"  sim hand pos={cur[7:]}")  # right hand current
            self.phase_step += 1
            fingers = self._compute_finger_targets(close=close_fingers, hand=self.hand, env=env)
            coarse_targets = torch.tensor(
                np.concatenate([arm, fingers]),
                device=self.device, dtype=torch.float32,
            )
            return coarse_targets, self._make_phase_intent()

        if self.phase == Phase.APPROACH:
            target_w = block_pos_world.copy()
            target_w[2] += 0.12  # approach from well above to avoid pushing block
            arm = self._solve_ik(self._world_to_robot(target_w), self.hand, env=env)
            if dist_to_block < self.config.approach_dist:
                self.phase = Phase.PRE_GRASP
                self.phase_step = 0

        elif self.phase == Phase.PRE_GRASP:
            arm = self._solve_ik(self._world_to_robot(block_pos_world), self.hand, env=env)
            self.phase_step += 1
            if self.phase_step >= PHASE_HOLD[Phase.PRE_GRASP]:
                self.phase = Phase.GRASP
                self.phase_step = 0

        elif self.phase == Phase.GRASP:
            arm = self._solve_ik(self._world_to_robot(block_pos_world), self.hand, env=env)
            close_fingers = True
            self.phase_step += 1
            if self.phase_step >= PHASE_HOLD[Phase.GRASP] or contact > self.config.grasp_force_threshold:
                self.phase = Phase.LIFT
                self.phase_step = 0

        elif self.phase == Phase.LIFT:
            lift_w = block_pos_world.copy()
            lift_w[2] += self.config.lift_height
            arm = self._solve_ik(self._world_to_robot(lift_w), self.hand, env=env)
            close_fingers = True
            if ee_pos_world[2] > block_pos_world[2] + self.config.lift_height * 0.8:
                self.phase = Phase.TRANSPORT
                self.phase_step = 0

        elif self.phase == Phase.TRANSPORT:
            transport_w = place_pos_world.copy()
            transport_w[2] += self.config.lift_height
            arm = self._solve_ik(self._world_to_robot(transport_w), self.hand, env=env)
            close_fingers = True
            dist_to_place = np.linalg.norm(ee_pos_world[:2] - place_pos_world[:2])
            if dist_to_place < self.config.approach_dist * 2:
                self.phase = Phase.DESCEND
                self.phase_step = 0

        elif self.phase == Phase.DESCEND:
            arm = self._solve_ik(self._world_to_robot(place_pos_world), self.hand, env=env)
            close_fingers = True
            height_error = abs(ee_pos_world[2] - place_pos_world[2])
            if height_error < 0.02:
                self.phase = Phase.RELEASE
                self.phase_step = 0

        elif self.phase == Phase.RELEASE:
            arm = self._solve_ik(self._world_to_robot(place_pos_world), self.hand, env=env)
            self.phase_step += 1
            if self.phase_step >= PHASE_HOLD[Phase.RELEASE]:
                self.phase = Phase.RETREAT
                self.phase_step = 0

        elif self.phase == Phase.RETREAT:
            retreat_w = place_pos_world.copy()
            retreat_w[2] += self.config.lift_height
            arm = self._solve_ik(self._world_to_robot(retreat_w), self.hand, env=env)
            self.phase_step += 1
            if self.phase_step >= PHASE_HOLD[Phase.RETREAT]:
                self.current_block_idx += 1
                self.phase = Phase.APPROACH
                self.phase_step = 0

        else:
            arm = self._current_arm_q

        # Step 5: Verbose logging every 50 steps or first step
        if self.phase_step % 50 == 0 or self.phase_step == 1:
            print(f"[Controller] phase={self.phase.name} hand={self.hand} "
                  f"block_idx={self.current_block_idx} step={self.phase_step} "
                  f"dist={dist_to_block:.4f} contact={contact:.3f}")

        # Retract fingers during approach/retreat to avoid pushing objects
        retract = (self.phase in (Phase.APPROACH, Phase.RETREAT)) and not close_fingers
        fingers = self._compute_finger_targets(close=close_fingers, hand=self.hand, env=env, retract=retract)
        coarse_targets = torch.tensor(
            np.concatenate([arm, fingers]),
            device=self.device, dtype=torch.float32,
        )
        return coarse_targets, self._make_phase_intent()

    def reset(self):
        self.current_block_idx = 0
        self.phase = Phase.APPROACH
        self.phase_step = 0
        self.hand = "right"
        self._current_arm_q = np.zeros(14)
        self._body_idx_cache = {}
        self._initialized_transform = False
        self._finger_limits_read = False
        self._first_step_done = False
