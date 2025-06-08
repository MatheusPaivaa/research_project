import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


log_dir = "./logs/curriculum_logs"
gif_path = os.path.join(log_dir, "unlocked_bins_evolution.gif")
mp4_path = os.path.join(log_dir, "unlocked_bins_evolution.mp4")
max_range = 3.0
step_size = 0.2
interval_ms = 100

num_bins = int((2 * max_range) / step_size) + 1
lin_vals = np.linspace(-max_range, max_range, num_bins)
ang_vals = np.linspace(-max_range, max_range, num_bins)
lin_to_idx = {round(v, 4): i for i, v in enumerate(lin_vals)}
ang_to_idx = {round(v, 4): i for i, v in enumerate(ang_vals)}

files = sorted([
    f for f in os.listdir(log_dir) if f.startswith("bins_step_") and f.endswith(".pkl")
], key=lambda x: int(x.split("_step_")[1].split(".")[0]))

grids = []
steps = []

for f in files:
    path = os.path.join(log_dir, f)
    with open(path, 'rb') as pf:
        unlocked_bins = pickle.load(pf)

    grid = np.zeros((num_bins, num_bins))
    for x, z in unlocked_bins:
        xi = lin_to_idx.get(round(x, 4))
        zi = ang_to_idx.get(round(z, 4))
        if xi is not None and zi is not None:
            grid[zi, xi] = 1

    grids.append(grid)
    steps.append(int(f.split("_step_")[1].split(".")[0]))

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(grids[0], cmap='Blues', origin='lower',
               extent=[-max_range, max_range, -max_range, max_range],
               vmin=0, vmax=1)

ax.set_xticks(np.linspace(-max_range, max_range, num_bins), minor=True)
ax.set_yticks(np.linspace(-max_range, max_range, num_bins), minor=True)
ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)

filled_inicial = int(np.sum(grids[0]))
title = ax.set_title(f"Step {steps[0]} - {filled_inicial} bins filled")
plt.xlabel("lin_vel_x")
plt.ylabel("ang_vel_z")

def update(i):
    im.set_data(grids[i])
    filled = int(np.sum(grids[i]))
    title.set_text(f"Step {steps[i]} - {filled} bins filled")
    return [im, title]

ani = animation.FuncAnimation(
    fig, update, frames=len(grids), interval=interval_ms, blit=False)

ani.save(gif_path, writer='pillow', fps=1000 // interval_ms)
print(f"GIF: {gif_path}")

try:
    ani.save(mp4_path, writer=animation.FFMpegWriter(fps=1000 // interval_ms))
    print(f"MP4 {mp4_path}")
except Exception as e:
    print(f"Error MP4: {e}")

final_grid = grids[-1]
total_bins = final_grid.size
filled = int(np.sum(final_grid))
blank_spaces = total_bins - filled
print(f"Blank spaces: {blank_spaces}")