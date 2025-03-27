import argparse
from config.anymal_c_config import get_train_cfg, get_cfgs

def main():
    """Main function to parse arguments and run training or evaluation."""

    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="anymal_c_walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=200)
    parser.add_argument("--train", action="store_true", default=False)
    args = parser.parse_args()

    # Setup Experiment
    #log_dir = setup_experiment(args)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Create Environment

    # Train or Evaluate

    if args.train:
        print("a")
    else:
        print("b")

if __name__ == "__main__":
    main()