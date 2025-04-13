from isaaclab.utils import configclass

scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from envs.env_setup import CreateEnv

@configclass
class AnymalCEnvCfg(CreateEnv):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

