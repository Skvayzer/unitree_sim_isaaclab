# System 0 Documentation Index

## Active Specifications

| File | Type | Description |
|------|------|-------------|
| [system0_project_spec.md](system0_project_spec.md) | Reference | Architecture, code map, obs/action spaces, reward, hyperparams, key numbers |
| [system0_training_status.md](system0_training_status.md) | Status | Current eval results, bug status, roadmap, MoDE-VLA gap analysis |
| [training_diary.md](training_diary.md) | Log | Chronological experiments, findings, and decisions |

## Research & Analysis

| File | Description |
|------|-------------|
| [mode_vla_analysis.md](mode_vla_analysis.md) | MoDE-VLA / IMCopilot / RECIPE / DPPO paper analysis |

## CraftNet Eval

| File | Description |
|------|-------------|
| [craftnet_split_eval_instructions.md](../../unitree_IL_lerobot/docs/craftnet_split_eval_instructions.md) | Split System 1/System 2 eval architecture — launch procedure, protocols, latency budget |

## Future Work Design Documents

| File | Status | Description |
|------|--------|-------------|
| [domain_randomization_design.md](domain_randomization_design.md) | Not implemented | DR ranges, EventTerm API, training strategy |
| [adaptation_module_design.md](adaptation_module_design.md) | Not implemented | PrivilegedEncoder + LSTM teacher-student distillation |
| [craftnet_integration_design.md](craftnet_integration_design.md) | Not implemented | Residual injection, tactile gate, hierarchical switching |

## IsaacSim Installation

| File | Description |
|------|-------------|
| [isaacsim5.1_install.md](isaacsim5.1_install.md) | IsaacSim 5.1 installation guide |
| [isaacsim5.0_install.md](isaacsim5.0_install.md) | IsaacSim 5.0 installation guide |
| [isaacsim4.5_install.md](isaacsim4.5_install.md) | IsaacSim 4.5 installation guide |

## Deprecated

The following specs in `.auto-claude/specs/` are **outdated** and should not be used:
- `001-train-system0-moe-block-stacking-policy/` — MoE approach abandoned
- `003-train-robot-grasping-and-block-stacking/` — Superseded by current specialist decomposition
