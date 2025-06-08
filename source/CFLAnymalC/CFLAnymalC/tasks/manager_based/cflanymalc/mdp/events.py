from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from terrain_stats import terrain_stats   

@configclass
class EventCfg:
    """Configuration for environmental and robot-related events in the simulation."""

    # Events triggered every simulation step

    # Event term that tracks per-terrain performance (reward, tracking error, episode length)
    terrain_perf = EventTerm(
        func = terrain_stats,
        mode = "step", # called every physics step
        params = dict(window=500, print_every=10_000)
    )

    # Events triggered once at environment startup (before any resets)

    # Randomizes the physical material properties (friction, restitution) of the robot's foot
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params=dict(
            asset_cfg=SceneEntityCfg("robot", body_names=".*FOOT"),
            static_friction_range=(0.4, 1.5),     
            dynamic_friction_range=(0.3, 1.3),
            restitution_range=(0.0, 0.5),
            num_buckets=32,
        ),
    )

    # Adds or subtracts mass from the robot's base to simulate mass uncertainty
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),  # +/- 5kg variation
            "operation": "add", 
        },
    )

    # Events triggered at every environment reset

    # Applies external forces/torques to the base during reset (can simulate pushes or perturbations)
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-10.0, 10.0),   
            "torque_range": (-5.0, 5.0),
        },
    )

    # Resets the robot base (position, orientation, velocity) within a random uniform range
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),  # full rotation allowed
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # Randomly scales the initial joint positions and resets joint velocities to zero
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),  # scales joint positions (e.g., 50% to 150%)
            "velocity_range": (0.0, 0.0),  # joints start with zero velocity
        },
    )

    # Events triggered periodically during simulation (in between steps)

    # Pushes the robot by directly setting linear velocity within a random range
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",  # triggered at random intervals
        interval_range_s=(10.0, 15.0),  # push every 10–15 seconds
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            },
        },
    )