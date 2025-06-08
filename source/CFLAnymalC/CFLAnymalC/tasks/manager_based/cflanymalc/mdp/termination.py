from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg

@configclass
class TerminationsCfg:
    """Termination conditions for the MDP."""

    # Terminate when the episode exceeds the maximum allowed time steps
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if the robot's base comes into contact with the ground (illegal contact)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )

    # Terminate if the robot's base orientation exceeds a tilt threshold (roll or pitch)
    large_body_tilt = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Terminate if the robot is out of terrain's bounds
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

    # Terminate if the applied joint effort (torque) hits the actuator's soft limits
    # joint_effort_limit = DoneTerm(
    #     func=mdp.joint_effort_out_of_limit,
    #     params=dict(
    #         asset_cfg=SceneEntityCfg("robot")
    #     ),
    # )

    # Terminate if joint velocities exceed the soft joint velocity limits
    # joint_vel_limit = DoneTerm(
    #     func=mdp.joint_vel_out_of_limit,
    #     params=dict(
    #         asset_cfg=SceneEntityCfg("robot")
    #     ),
    # )
