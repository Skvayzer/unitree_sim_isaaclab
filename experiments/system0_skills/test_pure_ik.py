#!/usr/bin/env python3
"""Test pure IK reaching — bypass System 1, send IK joints directly to sim via ZMQ.

Computes IK solution for the block target, then sends those joint positions
directly as actions to the eval pipeline. This tests whether the IK solver
CAN bring the hand to the block, independent of the DiT.
"""
import json
import time
import io
import numpy as np
import zmq
import msgpack
import msgpack_numpy as m
from PIL import Image

m.patch()

def compress_jpeg(img, quality=50):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=quality)
    return buf.getvalue()

def main():
    # Connect to System 1 (but we'll override its action)
    # Actually — System 1 runs the DiT. We can't bypass it from outside.
    # Instead: modify system1_server to have a "pure IK" mode.

    # Alternative: compute IK trajectory and report what SHOULD happen
    import sys, os
    sys.path.insert(0, os.path.expanduser('~/unitree_IL_lerobot'))
    from unitree_lerobot.eval_robot.ik_prior import IKPriorComputer

    ik = IKPriorComputer()

    # Get current state and block position
    d = json.load(open('/tmp/sim_block_positions.json'))
    block = np.array(d['red_block']['rel'])

    with open('/tmp/arm_debug.txt') as f:
        last = f.readlines()[-1]
    state_str = last.split('STATE:')[1].strip().strip('[]')
    current_14 = np.array([float(x) for x in state_str.split(',')])

    print(f"Block (robot-rel): {block.round(4)}")
    print(f"Current right arm: {current_14[7:14].round(3)}")
    _, r_ee = ik.fk(current_14)
    print(f"Current FK wrist:  {r_ee.round(4)}")
    print(f"Wrist→block:       {np.linalg.norm(r_ee - block)*100:.1f}cm")
    print()

    # Compute IK trajectory: 16 steps from current to target
    print("Computing IK trajectory...")
    traj = ik.compute(current_14, block, hand='R', n_steps=16)

    # Show trajectory
    for step in [0, 4, 8, 12, 15]:
        q = traj[step, :14]
        _, ee = ik.fk(q)
        dist = np.linalg.norm(ee - block) * 100
        print(f"  Step {step:2d}: R_arm={q[7:14].round(3)}  EE={ee.round(3)}  dist={dist:.1f}cm")

    # Final IK solution quality
    _, final_ee = ik.fk(traj[-1, :14])
    final_dist = np.linalg.norm(final_ee - block) * 100
    print(f"\nFinal IK distance: {final_dist:.1f}cm")
    print(f"Final IK Z error:  {(final_ee[2]-block[2])*100:.1f}cm")

    if final_dist < 5.0:
        print("IK CAN reach the block! The problem is in the DiT pipeline.")
    elif final_dist < 15.0:
        print("IK gets close but not precise. DiT should refine the last few cm.")
    else:
        print("IK CANNOT reach the block. Target is outside workspace or frame is wrong.")

    # Now send IK joints directly as actions (bypass DiT)
    # Connect to System 1 and send a special command
    print("\n=== Sending pure IK joints to sim ===")
    print("(Connecting to System 1 at localhost:5556)")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 10000)
    sock.connect('tcp://127.0.0.1:5556')

    # The IK trajectory gives us target joint positions
    # We need to send them as the 28D action
    # Action format: [left_arm(7), right_arm(7), left_hand(7), right_hand(7)]

    target_action = np.zeros(28, dtype=np.float32)
    target_action[:7] = current_14[:7]  # left arm: stay at current
    target_action[7:14] = traj[-1, 7:14]  # right arm: IK solution
    # fingers: zeros (open hand)

    print(f"Sending action - right arm: {target_action[7:14].round(3)}")

    black = np.zeros((480, 640, 3), dtype=np.uint8)
    jpeg = compress_jpeg(black)

    # Send repeatedly for 30 seconds
    for i in range(300):
        req = {
            'type': 'predict',
            'images': {'observation.images.cam_left_high': jpeg},
            'state': current_14.tolist() + [0]*14,  # 28D state
            'task': 'pick up the red block',
        }
        sock.send(msgpack.packb(req, use_bin_type=True))
        resp = msgpack.unpackb(sock.recv(), raw=False)

        # Check distance every second
        if i % 10 == 0:
            d2 = json.load(open('/tmp/sim_block_positions.json'))
            sim_ee = np.array(d2['right_ee']['rel'])
            bp = np.array(d2['red_block']['rel'])
            dist = np.linalg.norm(sim_ee - bp) * 100
            print(f"  [{i/10:.0f}s] Palm→block: {dist:.1f}cm  palm={sim_ee.round(3)}")

        time.sleep(0.1)

    sock.close()
    ctx.term()

if __name__ == '__main__':
    main()
