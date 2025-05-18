import json
import numpy as np
import matplotlib.pyplot as plt

flat_flat_path = "./logs/tracking_log/average_error_log_flat_in_wave.json"

def load_error_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    v_x_vals = sorted(set(d["v_x_cmd"] for d in data))
    omega_z_vals = sorted(set(d["omega_z_cmd"] for d in data))

    error_x = np.zeros((len(omega_z_vals), len(v_x_vals)))
    error_omega = np.zeros((len(omega_z_vals), len(v_x_vals)))

    for d in data:
        i = omega_z_vals.index(d["omega_z_cmd"])
        j = v_x_vals.index(d["v_x_cmd"])
        error_x[i, j] = d["avg_error_x"]
        error_omega[i, j] = d["avg_error_omega_z"]

    return error_x, error_omega, v_x_vals, omega_z_vals

# Load data
error_x_flat, error_omega_flat, v_x_vals, omega_z_vals = load_error_data(flat_flat_path)

vmin = 0.005
vmax = 3
cmap = plt.get_cmap("Blues_r")

# Adjust to 2 rows and 1 column
fig, axs = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'hspace': 0.3})
fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.92)

titles = ["Flat"]

# Plot error_x
im1 = axs[0].imshow(error_x_flat, origin="lower", cmap=cmap,
                    extent=[v_x_vals[0], v_x_vals[-1], omega_z_vals[0], omega_z_vals[-1]],
                    aspect='equal', vmin=vmin, vmax=vmax)
axs[0].set_title("Average Error X (Flat)", fontname='DejaVu Serif')
axs[0].set_ylabel("$\\omega_z$ [rad/s]", fontname='DejaVu Serif')
axs[0].grid(alpha=0.2)

# Plot error_omega
im2 = axs[1].imshow(error_omega_flat, origin="lower", cmap=cmap,
                    extent=[v_x_vals[0], v_x_vals[-1], omega_z_vals[0], omega_z_vals[-1]],
                    aspect='equal', vmin=vmin, vmax=vmax)
axs[1].set_title("Average Error $\\omega_z$ (Flat)", fontname='DejaVu Serif')
axs[1].set_xlabel("$v_x$ [m/s]", fontname='DejaVu Serif')
axs[1].set_ylabel("$\\omega_z$ [rad/s]", fontname='DejaVu Serif')
axs[1].grid(alpha=0.2)

# Add colorbar
fig.colorbar(im2, ax=axs, orientation='vertical', fraction=0.025, pad=0.04)

fig.tight_layout(pad=1.5)
plt.show()
