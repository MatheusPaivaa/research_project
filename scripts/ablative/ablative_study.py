import subprocess

terrains = ["pyramid", "pyramid_inv"]

training_commands_flat = [
    f"./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/train.py --task CFL-AnymalC_Flat_Task --terrain=flat --headless"
]

training_commands_rough = [
    f"./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Rough_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_rough/generalist/model_3999.pt --trained_terrain=generalist --tested_terrain=" + terrain + " --log_name=generalist_lstm_" + terrain +  " --headless"
    for terrain in terrains
]

training = training_commands_flat + training_commands_rough

testing_terrains = ["flat", "waves", "boxes", "stepping_stones", "flat_oil"]

all_testing_commands = [
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Rough_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_rough/generalist/model_3999.pt --trained_terrain=generalist --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_flat/flat/model_3999.pt --trained_terrain=flat --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Rough_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_rough/flat_oil/model_3999.pt --trained_terrain=flat_oil --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Rough_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_rough/waves/model_3999.pt --trained_terrain=waves --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Rough_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_rough/boxes/model_3999.pt --trained_terrain=boxes --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Rough_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_rough/stepping_stones/model_3999.pt --trained_terrain=stepping_stones --tested_terrain=" + terrain + " --headless"
        for terrain in testing_terrains
    ]
]

new_terrains_testing = ["pyramid", "pyramid_inv"]

testing_different_terrains_generalist = [
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Rough_Task --num_envs 150 --checkpoint ./logs/rsl_rl_inicial/anymal_c_rough/generalist/model_3999.pt --trained_terrain=generalist --tested_terrain=" + terrain + " --headless"
        for terrain in new_terrains_testing
    ]
]

testing_different_terrains_specific = [
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Rough_Task --num_envs 150 --checkpoint ./logs/rsl_rl_new_terrains/anymal_c_rough/" + terrain + "/model_3999.pt --trained_terrain=" + terrain + " --tested_terrain=" + terrain + " --headless"
        for terrain in new_terrains_testing
    ]
]

testing_curriculum = [
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/step_02_005_015/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --log_name=step_02_005_015 --headless"
    ], 
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/step_05_005_015/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --log_name=step_05_005_015 --headless"
    ], 
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/step_05_005_015/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --log_name=step_05_005_015 --headless"
    ], 
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/steps_05_015_025/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --log_name=steps_05_015_025 --headless"
    ], 
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/log_decay_step_02/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --log_name=log_decay_step_02 --headless"
    ], 
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/log_decay_step_05/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --log_name=log_decay_step_05 --headless"
    ]
]

testing_curriculum = [
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/rsl_rl_035/anymal_c_flat/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --log_name=increment_035 --headless"
    ], 
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/rsl_rl_sem_redundancia/anymal_c_flat/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --log_name=sem_redundancia_05 --headless"
    ], 
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/rsl_rl_sem_regresso/anymal_c_flat/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --log_name=sem_regresso_05 --headless"
    ], 
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/rsl_rl_sem_regresso_unlocked/anymal_c_flat/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --log_name=sem_regresso_unlocked_05 --headless"
    ]
]


testing_curriculum = [
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_flat/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --headless --log_name=janela_media"
    ], 
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/rsl_rl_sem_regresso/anymal_c_flat/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --headless --log_name=sem_regresso"
    ],
    *[
        "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task CFL-AnymalC_Play_Flat_Task --num_envs 150 --checkpoint ./logs/rsl_rl_janela_antes/anymal_c_flat/flat/model_3999.pt --trained_terrain=flat --tested_terrain=flat --headless --log_name=janela_delay"
    ]
]


for cmd in (testing_curriculum):
    print(f"Running: {cmd}")
    process = subprocess.run(cmd, shell=True)
    
    if process.returncode != 0:
        print(f"Error while executing the command: {cmd}")
        break  
    else:
        print(f"Command completed successfully: {cmd}\n")
