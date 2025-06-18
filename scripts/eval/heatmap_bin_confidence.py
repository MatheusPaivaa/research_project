import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

log_dir = "./logs/curriculum_logs"
gif_path = os.path.join(log_dir, "bin_confidence_evolution.gif")
mp4_path = os.path.join(log_dir, "bin_confidence_evolution.mp4")

# Configuration for the bin grid
max_range = 3.0
step_size = 0.2
interval_ms = 100

# Compute the number of bins per dimension and value arrays
num_bins = int((2 * max_range) / step_size) + 1
lin_vals = np.linspace(-max_range, max_range, num_bins)
ang_vals = np.linspace(-max_range, max_range, num_bins)

# Mapping values to their corresponding grid indices
lin_to_idx = {round(v, 4): i for i, v in enumerate(lin_vals)}
ang_to_idx = {round(v, 4): i for i, v in enumerate(ang_vals)}

# Load all bin confidence files and sort them by training step
files = sorted([
    f for f in os.listdir(log_dir) if f.startswith("bins_confidence") and f.endswith(".pkl")
], key=lambda x: int(x.split("_step_")[1].split(".")[0]))

grids = []
steps = []

# Read and store each confidence grid over time
for f in files:
    path = os.path.join(log_dir, f)
    with open(path, 'rb') as pf:
        bin_conf = pickle.load(pf)

    grid = np.zeros((num_bins, num_bins))

    # Fill the grid with confidence values at appropriate indices
    for (x, z), confidence in bin_conf.items():
        xi = lin_to_idx.get(round(x, 4))
        zi = ang_to_idx.get(round(z, 4))
        if xi is not None and zi is not None:
            grid[zi, xi] = confidence

    grids.append(grid)
    steps.append(int(f.split("_step_")[1].split(".")[0]))

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(grids[0], cmap='Blues', origin='lower',
               extent=[-max_range, max_range, -max_range, max_range],
               vmin=0, vmax=1)

ax.set_xticks(np.linspace(-max_range, max_range, num_bins), minor=True)
ax.set_yticks(np.linspace(-max_range, max_range, num_bins), minor=True)
ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)

title = ax.set_title(f"Step {steps[0]}")
plt.xlabel("lin_vel_x")
plt.ylabel("ang_vel_z")

# Frame update function for animation
def update(i):
    im.set_data(grids[i])
    title.set_text(f"Step {steps[i]}")
    return [im, title]

ani = animation.FuncAnimation(
    fig, update, frames=len(grids), interval=interval_ms, blit=False)

ani.save(gif_path, writer='pillow', fps=1000 // interval_ms)
print(f"GIF saved to: {gif_path}")

try:
    ani.save(mp4_path, writer=animation.FFMpegWriter(fps=1000 // interval_ms))
    print(f"MP4 saved to: {mp4_path}")
except Exception as e:
    print(f"Failed to save MP4: {e}")