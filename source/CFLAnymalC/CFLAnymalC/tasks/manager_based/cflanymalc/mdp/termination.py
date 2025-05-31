from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
