import json
import numpy as np
import matplotlib.pyplot as plt

flat_flat_path = "./logs/tracking_log/average_error_log_flat_in_flat.json"
flat_wave_path = "./logs/tracking_log/average_error_log_flat_in_wave.json"
flat_stepping_path = "./logs/tracking_log/average_error_log_flat_in_stepping.json"

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

error_x_flat, error_omega_flat, v_x_vals, omega_z_vals = load_error_data(flat_flat_path)
error_x_wave, error_omega_wave, _, _ = load_error_data(flat_wave_path)
error_x_stepping, error_omega_stepping, _, _ = load_error_data(flat_stepping_path)

vmin = 0.005
vmax = 1
cmap = plt.get_cmap("Blues_r")

fig, axs = plt.subplots(2, 3, figsize=(14, 6), gridspec_kw={'wspace': 0, 'hspace': 0.15})
fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.85)
titles = ["Flat", "Wave", "Stepping Stones"]

for col, (error, title) in enumerate(zip([error_x_flat, error_x_wave, error_x_stepping], titles)):
    im = axs[0, col].imshow(error, origin="lower", cmap=cmap,
                            extent=[v_x_vals[0], v_x_vals[-1], omega_z_vals[0], omega_z_vals[-1]],
                            aspect='equal', vmin=vmin, vmax=vmax)
    axs[0, col].tick_params(labelbottom=False)
    axs[0, col].grid(alpha=0.2)
    if col != 0:
        axs[0, col].tick_params(labelleft=False)
    if title == "Flat":
        axs[0, col].set_title(title, fontname='DejaVu Serif')
    else:
        axs[0, col].set_title(title, fontname='DejaVu Serif') 

    for spine in axs[0, col].spines.values():
        spine.set_linewidth(1.3)
        axs[0, col].tick_params(width=1.3, length=6)

    axs[0, col].spines['right'].set_visible(False)
    axs[0, col].spines['top'].set_visible(False)

for col, (error, title) in enumerate(zip([error_omega_flat, error_omega_wave, error_omega_stepping], titles)):
    im = axs[1, col].imshow(error, origin="lower", cmap=cmap,
                            extent=[v_x_vals[0], v_x_vals[-1], omega_z_vals[0], omega_z_vals[-1]],
                            aspect='equal', vmin=vmin, vmax=vmax)
    axs[1, col].tick_params(labelleft=False)
    if col == 0:
        axs[1, col].set_xlabel("$v_x$ [m/s]", fontname='DejaVu Serif')
        axs[1, col].set_ylabel("$\\omega_z$ [rad/s]", fontname='DejaVu Serif')
        axs[1, col].tick_params(labelleft=True) 
    axs[1, col].grid(alpha=0.2)

    axs[1, col].tick_params(pad=10, width=2, length=6)

    for spine in axs[1, col].spines.values():
        spine.set_linewidth(1.3)
        axs[1, col].tick_params(width=1, length=6)

    axs[1, col].spines['right'].set_visible(False)
    axs[1, col].spines['top'].set_visible(False)

cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical', fraction=0.025, pad=0.02)
cbar.outline.set_linewidth(1.3)
cbar.ax.tick_params(width=1.2, length=6)

fig.tight_layout(pad=1.5)
plt.show()
