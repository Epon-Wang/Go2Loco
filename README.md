# ROB-GY 6323 Go2 Project

Training a robust gait policy for Unitree Go2 quadrupedal locomotion using Proximal Policy Gradient (PPO)

[Collaborators](#collaborators)\
[Installation](#installation)\
[Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Notes](#notes)

[Implementation](#implementation)
- [MDP](#mdp)
    - [Reward](#reward)
    - [Termination](#termination)
    - [Observation](#observation)
- [PD Controller](#low-level-pd-controller)
- [Actuator Friction Model](#actuator-friction-model)


## Collaborators

- [Zeyu Jiang](https://github.com/JDScript)
- [Yipeng Wang](https://github.com/Epon-Wang)

## Installation

```bash
git submodule update --init --recursive
uv sync --all-groups
```

## Usage

```bash
cd rob6323_go2_project
source ./.venv/bin/activate
```


### Training

To train in headless mode

```bash
python ./scripts/rsl_rl/train.py \
--task=<task_name> \
--headless
```

To train in livestream mode

```bash
python ./scripts/rsl_rl/train.py \
--task=<task_name> \
--livestream 2
```

Then you can watch the rendered training with [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)

### Evaluation

To visualize policy rollout in 16 parallel environments

```bash
python ./scripts/rsl_rl/play.py \
--task=<task_name> \
--checkpoint /path/to/ckpt.pt \
--num_envs 16
```

To record a video for policy rollout

```bash
python ./scripts/rsl_rl/play.py \
--task=<task_name> \
--checkpoint /path/to/ckpt.pt \
--video \
--video_length 2000 \
--headless
```

### Notes

- Supported tasks are

    | `<task_name>` | Task Description |
    |-------|-------------|
    |`Template-Rob6323-Go2-Direct-v0`   | Gait Policy Baseline |
    |`Template-Rob6323-Go2-Direct-v1`     | Gait Policy with Actuator Friction Model |

- Agent & Environment seed could be specified AT THE SAME TIME with an optional flag `--seed <seed_num>`
- Training logs, checkpoints, and rollout videos could be found in the `logs` directory


## Implementation

### MDP

For **rewards** and **terminations**, please refer to [MDP](./source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/mdp/__init__.py) directory for implementation

#### Reward

| Term            | Weight    | Description |
|:-------           |:------:   |:-------------|
|`track_cmd_linVel` | 1.0       | on tracking commands of body xy linear velocity |
|`track_cmd_angVel` | 0.5       | on tracking commands of body z(yaw) angular velocity |
|`body_orient`      |-5.0       | on keeping body orientation upright |
|`body_pose`        |-0.001     | on minimal body rolling & pitching |
|`dof_vel`          |-0.0001    | on minimal joint velocity |
|`dof_torque`       |-0.00005   | on minimal joint torque |
|`action_rate`      |-0.1       | on action rate |
|`bouncing`         |-0.02      | on minimal bouncing |
|`raibert_heuristic`|-10.0      | on Raibert Heuristic gait shapping |
|`foot_clearance`   |-30.0      | on lifting feet during swing phase |
|`tracking_contacts`|4.0        | on tracking foot contacts states |

>Weights of `dof_torque`/`action_rate` are tuned to -0.00002/-0.05 in the task with actuator friction model

#### Termination

| Term                  | Description |
|:-------               |:-------------|
| `done_timeout`        | Episode length has reached its maximum |
| `done_baseContact`    | Any contact force on robot base exceeds 1.0N threshold |
| `done_baseHeight`     | Height of robot base lower than 25cm threshold |
| `done_baseUpsidedown` | Robot base upsidedown |

#### Observation

| Term | Description |
|:-------  |:-------------|
| `root_lin_vel_b` | Root COM linear velocity in robot base frame |
| `root_ang_vel_b` | root COM linear velocity in robot base frame |
| `projected_gravity_b` | projection of gravity direction on robot base frame |
| `_commands` | task command |
| `joint_pos - default_joint_pos` | change in robot joint position from default value |
| `joint_vel` | robot joint velocity |
| `_actions` | robot actions
| `clock_inputs` | gait clock of Raibert Heuristics |

### Low-level PD Controller

Joint positions are computed from policy actions, and joint torques are computed from joint positions

$$
\tau = K_p(q_{des} - q) - K_d\dot{q}
$$

### Actuator Friction Model

For the convenience of sim2real, this model computes static and viscous friction that should be subtracted from the torque computed by PD Controller

$$
\begin{align*}
\tau_{stiction} &= F_s \cdot \tanh(\frac{\dot{q}}{0.1})\\
\tau_{viscous} &= \mu_v \cdot \dot{q}\\
\tau_{fraction} &= tau_{stiction} + \tau_{viscous}\\
\tau_{PD} &\leftarrow \tau_{PD} - \tau_{friction}
\end{align*}
$$

where stiction coefficient $F_{s}$ and viscous coefficient $\mu_v$ are randomized per reset of episode $\epsilon$ with

$$
\mu_v^{\epsilon} \sim \mathcal{U}(0.0, 0.3) \quad \quad F_{s}^{\epsilon} \sim \mathcal{U}(0.0, 2.5)
$$