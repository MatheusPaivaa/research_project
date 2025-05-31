import subprocess

terrains = ["all", "boxes", "flat_oil", "boxes", "stepping_stones", "waves"]

training_commands_flat = [
    f"./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/train.py --task CFL-AnymalC_Flat_Task --terrain=flat --headless"
]

training_commands_rough = [
    f"./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/train.py --task CFL-AnymalC_Rough_Task --terrain={terrain} --headless"
    for terrain in terrains
]

training = training_commands_flat + training_commands_rough

testing_terrains = ["flat", "waves", "boxes", "stepping_stones", "flat_oil"]

all_testing_commands_default = [
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Default_Task --num_envs 150 --checkpoint ./logs/rsl_rl_default/anymal_c_default_direct/generalist/model_1499.pt --trained_terrain=generalist --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Raw_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_default/flat/model_3999.pt --trained_terrain=flat --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Default_Task --num_envs 150 --checkpoint ./logs/rsl_rl_default/anymal_c_default_direct/flat_oil/model_1499.pt --trained_terrain=flat_oil --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Default_Task --num_envs 150 --checkpoint ./logs/rsl_rl_default/anymal_c_default_direct/waves/model_1499.pt --trained_terrain=waves --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Default_Task --num_envs 150 --checkpoint ./logs/rsl_rl_default/anymal_c_default_direct/boxes/model_1499.pt --trained_terrain=boxes --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Default_Task --num_envs 150 --checkpoint ./logs/rsl_rl_default/anymal_c_default_direct/stepping_stones/model_1499.pt --trained_terrain=stepping_stones --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ]
]

for cmd in training:
    print(f"Running: {cmd}")
    process = subprocess.run(cmd, shell=True)
    
    if process.returncode != 0:
        print(f"Error while executing the command: {cmd}")
        break  
    else:
        print(f"Command completed successfully: {cmd}\n")
