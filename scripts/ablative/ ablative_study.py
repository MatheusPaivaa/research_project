import subprocess

terrains = ["flat", "all", "waves", "flat_oil", "boxes", "stepping_stones"]

training_commands = [
    f"./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/train.py --task CFL-AnymalC_Default_Task --terrain={terrain} --headless"
    for terrain in terrains
]

testing_terrains = ["flat", "waves", "boxes", "stepping_stones", "flat_oil"]

all_testing_commands = [
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Default_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_default_direct/generalist/model_1499.pt --trained_terrain=generalist --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Default_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_default_direct/flat/model_1499.pt --trained_terrain=flat --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Default_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_default_direct/flat_oil/model_1499.pt --trained_terrain=flat_oil --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Default_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_default_direct/waves/model_1499.pt --trained_terrain=waves --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Default_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_default_direct/boxes/model_1499.pt --trained_terrain=boxes --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Default_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_default_direct/stepping_stones/model_1499.pt --trained_terrain=stepping_stones --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ]
]

for cmd in all_testing_commands:
    print(f"Running: {cmd}")
    process = subprocess.run(cmd, shell=True)
    
    if process.returncode != 0:
        print(f"Error while executing the command: {cmd}")
        break  
    else:
        print(f"Command completed successfully: {cmd}\n")
