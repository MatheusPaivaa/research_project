import subprocess

# Lista de comandos a serem executados
comandos = [
    "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task Isaac-Anymal-C-custom --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_rough_direct/boxes/model_1499.pt --trained boxes --env wave --headless",
    "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task Isaac-Anymal-C-custom --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_rough_direct/generalist/model_1499.pt --trained generalist --env wave --headless", 
    "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task Isaac-Anymal-C-custom --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_rough_direct/wave/model_1499.pt --trained wave --env wave --headless",
    "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task Isaac-Anymal-C-custom --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_rough_direct/stepping/model_1499.pt --trained stepping --env wave --headless",
    "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task Isaac-Anymal-C-custom --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_rough_direct/oil/model_1499.pt --trained oil --env wave --headless",
    "./lib/IsaacLab/isaaclab.sh -p ./scripts/rsl_rl/play.py --task Isaac-Flat-Anymal-C-custom --num_envs 150 --checkpoint ./logs/rsl_rl/anymal_c_flat_direct/flat/model_499.pt --trained flat --env wave --headless"
]

# Executa cada comando sequencialmente
for cmd in comandos:
    print(f"Executando: {cmd}")
    processo = subprocess.run(cmd, shell=True)
    
    if processo.returncode != 0:
        print(f"Erro ao executar o comando: {cmd}")
        break  # Interrompe se algum comando falhar
    else:
        print(f"Comando finalizado com sucesso: {cmd}\n")
