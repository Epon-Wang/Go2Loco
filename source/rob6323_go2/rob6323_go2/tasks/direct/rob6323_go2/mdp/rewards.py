import torch
import isaaclab.utils.math as math_utils

from typing import TYPE_CHECKING
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv



def reward_trackCMD_linVel(
        env:    "DirectRLEnv",
        robot:  Articulation
    ) -> torch.Tensor:
    """
    reward on tracking commands of body xy linear velocity

    Input:
        - env:      Task Environment Instance
        - robot:    Robot Instance
    Output:
        - reward:   L2 distance between command & actual body linear velocity
    """
    lin_vel_error = torch.sum(
        torch.square(env._commands[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1,)
    return torch.exp(-lin_vel_error / 0.25)


def reward_trackCMD_angVel(
        env:    "DirectRLEnv",
        robot:  Articulation
    ) -> torch.Tensor:
    """
    reward on tracking commands of body z(yaw) angular velocity

    Input:
        - env:      Task Environment Instance
        - robot:    Robot Instance
    Output:
        - reward:   squared loss between command & actual body angular velocity
    """
    yaw_rate_error = torch.square(env._commands[:, 2] - robot.data.root_ang_vel_b[:, 2])
    return torch.exp(-yaw_rate_error / 0.25)


def reward_actionRate(
        env:            "DirectRLEnv",
        actionScale:    float
    ) -> torch.Tensor:
    """
    reward on action rate (1st & 2nd derivative of actions)

    Input:
        - env:          Task Environment Instance
        - actionScale:  action scale factor
    Output:
        - reward:       sum of L2 loss of 1st & 2nd derivative of actions
    """
    a = env._actions
    ap = env._last_actions[:, :, 0]
    app = env._last_actions[:, :, 1]
    # 1st Derivative
    # da = a[t] - a[t-1]
    first = torch.sum(torch.square(a - ap), dim=1) * (actionScale**2)
    # 2nd derivative
    # dda = da[t] - da[t-1] = a[t] - 2*a[t-1] + a[t-2]
    second = torch.sum(torch.square(a - 2*ap + app),dim=1) * (actionScale**2)
    return first + second


def reward_bodyOrient(
        robot:  Articulation
    ) -> torch.Tensor:
    """
    reward on keeping body orientation upright

    Input:
        - robot:    Robot Instance
    Output:
        - reward:   L2 loss on gravity projection onto x & y axis
    """
    return torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)


def reward_bodyPose(
        robot:  Articulation
    ) -> torch.Tensor:
    """
    reward on zero body rolling & pitching

    Input:
        - robot:    Robot Instance
    Output:
        - reward:   L2 loss on body angular velocity at roll & pitch 
    """
    return torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)


def reward_dofVel(
        robot:  Articulation
    ) -> torch.Tensor:
    """
    reward minimal joint velocity

    Input:
        - robot:    Robot Instance
    Output:
        - reward:   L2 loss on robot joint velocities
    """
    return torch.sum(torch.square(robot.data.joint_vel), dim=1)


def reward_bouncing(
        robot:  Articulation
    ) -> torch.Tensor:
    """
    reward on minimal bouncing

    Input:
        - robot:    Robot Instance
    Output:
        - reward:   squared loss on body z axis linear velocity
    """
    return torch.square(robot.data.root_lin_vel_b[:, 2])


def reward_raibertHeuristic(
        env:    "DirectRLEnv", 
        robot:  Articulation
    ) -> torch.Tensor:
    """
    Raibert Heuristic gait shapping

    Input:
        - env:      Task Environment Instance
        - robot:    Robot Instance
    Output:
        - reward:   L2 distance between desired & actual footsteps
    """
    cur_footsteps_translated = (
        env.foot_positions_w - robot.data.root_pos_w.unsqueeze(1)
    )
    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
            math_utils.quat_conjugate(robot.data.root_quat_w),
            cur_footsteps_translated[:, i, :],
        )

    # nominal positions: [FR, FL, RR, RL]
    desired_stance_width = 0.25
    desired_ys_nom = torch.tensor(
        [
            desired_stance_width / 2,
            -desired_stance_width / 2,
            desired_stance_width / 2,
            -desired_stance_width / 2,
        ],
        device=env.device,
    ).unsqueeze(0)

    desired_stance_length = 0.45
    desired_xs_nom = torch.tensor(
        [
            desired_stance_length / 2,
            desired_stance_length / 2,
            -desired_stance_length / 2,
            -desired_stance_length / 2,
        ],
        device=env.device,
    ).unsqueeze(0)

    # raibert offsets
    phases = torch.abs(1.0 - (env.foot_indices * 2.0)) * 1.0 - 0.5
    frequencies = torch.tensor([3.0], device=env.device)
    x_vel_des = env._commands[:, 0:1]
    yaw_vel_des = env._commands[:, 2:3]
    y_vel_des = yaw_vel_des * desired_stance_length / 2
    desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
    desired_ys_offset[:, 2:4] *= -1
    desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

    desired_ys_nom = desired_ys_nom + desired_ys_offset
    desired_xs_nom = desired_xs_nom + desired_xs_offset

    desired_footsteps_body_frame = torch.cat(
        (desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2
    )

    err_raibert_heuristic = torch.abs(
        desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2]
    )

    return torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
