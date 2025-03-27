import argparse
import os
import pickle
import shutil
import torch

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

def setup_experiment(args):
    """Initializes logging and configuration settings."""

    gs.init(logging_level="warning")
    
    log_dir = f"logs/{args.exp_name}"

    if args.train and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    return log_dir