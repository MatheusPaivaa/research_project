# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to simulate sand using particles in Isaac Lab.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/sand_simulation.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# Cria argparser
parser = argparse.ArgumentParser(description="Simulação de areia com partículas.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Lança o Isaac Sim com GUI
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Restante do código da simulação."""

from isaaclab.sim import SimulationCfg, SimulationContext
from omni.physx.scripts import particle_system
from omni.isaac.core.utils.prims import create_prim
from pxr import Usd

def main():
    """Função principal"""

    # Configura e inicia o contexto de simulação
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    # Define posição da câmera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Cria plano de chão
    create_prim("/World/ground", "Plane", position=(0, 0, 0), scale=(10, 10, 1))

    # Cria recipiente para conter a areia
    create_prim("/World/container", "Cube", position=(0, 0, 0.5), scale=(1.5, 1.5, 1),
                attributes={"physics:collisionEnabled": True})

    # Cria sistema de partículas
    ps_path = "/World/sand_particles"
    particle_system.create_particle_system(ps_path, max_particles=5000)

    # Cria emissor de areia (partículas físicas)
    emitter_path = f"{ps_path}/sand_emitter"
    particle_system.create_particle_emitter(
        emitter_path=emitter_path,
        particle_system_path=ps_path,
        emitter_type="omni.physx.particleSystem.sphereEmitter",
        position=(0, 0, 2),         # posição da "queda"
        count_rate=300,            # taxa de emissão
        radius=0.1,                # raio da emissão
        velocity=(0, 0, -2),       # velocidade inicial
        lifetime=5.0               # vida das partículas
    )

    # Ajusta o tamanho das partículas
    stage = sim.get_stage()
    ps_prim = stage.GetPrimAtPath(ps_path)
    ps_prim.GetAttribute("particleSize").Set(0.02)

    # Inicia simulação
    sim.reset()
    print("[INFO]: Simulação de areia iniciada...")

    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()
