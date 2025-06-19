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
