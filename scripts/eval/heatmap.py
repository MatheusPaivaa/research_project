import json
import numpy as np
import matplotlib.pyplot as plt

log_files = [
    {"name": "Generalist → Flat", "path": "./logs/tracking_log/average_error_log_generalist_in_flat.json"},
    {"name": "Generalist → Wave", "path": "./logs/tracking_log/average_error_log_generalist_in_waves.json"},
    {"name": "Generalist → Boxes", "path": "./logs/tracking_log/average_error_log_generalist_in_boxes.json"},
    {"name": "Generalist → Stepping", "path": "./logs/tracking_log/average_error_log_generalist_in_stepping_stones.json"},
    {"name": "Generalist → Oil Flat", "path": "./logs/tracking_log/average_error_log_generalist_in_flat_oil.json"},
]

# log_files = [
#     {"name": "Flat → Flat", "path": "./logs/tracking_log/average_error_log_flat_in_flat.json", "b": True},
#     {"name": "Flat → Wave", "path": "./logs/tracking_log/average_error_log_flat_in_waves.json"},
#     {"name": "Flat → Boxes", "path": "./logs/tracking_log/average_error_log_flat_in_boxes.json"},
#     {"name": "Flat → Stepping", "path": "./logs/tracking_log/average_error_log_flat_in_stepping_stones.json"},
#     {"name": "Flat → Oil Flat", "path": "./logs/tracking_log/average_error_log_flat_in_flat_oil.json"},
# ]

# log_files = [
#     {"name": "Wave → Flat", "path": "./logs/tracking_log/average_error_log_wave_in_flat.json"},
#     {"name": "Wave → Wave", "path": "./logs/tracking_log/average_error_log_wave_in_waves.json", "b": True},
#     {"name": "Wave → Boxes", "path": "./logs/tracking_log/average_error_log_wave_in_boxes.json"},
#     {"name": "Wave → Stepping", "path": "./logs/tracking_log/average_error_log_wave_in_stepping_stones.json"},
#     {"name": "Wave → Oil Flat", "path": "./logs/tracking_log/average_error_log_wave_in_flat_oil.json"},
# ]

# log_files = [
#     {"name": "Boxes → Flat", "path": "./logs/tracking_log/average_error_log_boxes_in_flat.json"},
#     {"name": "Boxes → Wave", "path": "./logs/tracking_log/average_error_log_boxes_in_waves.json"},
#     {"name": "Boxes → Boxes", "path": "./logs/tracking_log/average_error_log_boxes_in_boxes.json", "b": True},
#     {"name": "Boxes → Stepping Stones", "path": "./logs/tracking_log/average_error_log_boxes_in_stepping_stones.json"},
#     {"name": "Boxes → Oil Flat", "path": "./logs/tracking_log/average_error_log_boxes_in_flat_oil.json"},
# ]

# log_files = [
#     {"name": "Stepping → Flat", "path": "./logs/tracking_log/average_error_log_stepping_stones_in_flat.json"},
#     {"name": "Stepping → Wave", "path": "./logs/tracking_log/average_error_log_stepping_stones_in_waves.json"},
#     {"name": "Stepping → Boxes", "path": "./logs/tracking_log/average_error_log_stepping_stones_in_boxes.json"},
#     {"name": "Stepping → Stepping", "path": "./logs/tracking_log/average_error_log_stepping_stones_in_stepping_stones.json", "b": True},
#     {"name": "Stepping → Oil Flat", "path": "./logs/tracking_log/average_error_log_stepping_stones_in_flat_oil.json"},
# ]

# log_files = [
#     {"name": "Oil Flat → Flat", "path": "./logs/tracking_log/average_error_log_flat_oil_in_flat.json"},
#     {"name": "Oil Flat → Wave", "path": "./logs/tracking_log/average_error_log_flat_oil_in_waves.json"},
#     {"name": "Oil Flat → Boxes", "path": "./logs/tracking_log/average_error_log_flat_oil_in_boxes.json"},
#     {"name": "Oil Flat → Stepping", "path": "./logs/tracking_log/average_error_log_flat_oil_in_stepping_stones.json"},
#     {"name": "Oil Flat → Oil Flat", "path": "./logs/tracking_log/average_error_log_flat_oil_in_flat_oil.json", "b": True},
# ]


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

all_error_x = []
all_error_omega = []

v_x_vals, omega_z_vals = None, None 

general_errors = []

for log in log_files:
    error_x, error_omega, vx, wz = load_error_data(log["path"])
    all_error_x.append(error_x)
    all_error_omega.append(error_omega)

    combined_error = np.mean(np.sqrt(error_x**2 + error_omega**2))
    general_errors.append((log["name"], combined_error))

    if v_x_vals is None:
        v_x_vals, omega_z_vals = vx, wz  

print("\nErro geral por terreno:")
for name, err in general_errors:
    print(f"{name}: {err:.4f}")

vmin = 0.005
vmax = 3
cmap = plt.get_cmap("Blues_r")

fig, axs = plt.subplots(2, len(log_files), figsize=(14, 6), gridspec_kw={'wspace': 0.12, 'hspace': 0.05})
fig.subplots_adjust(top=0.79, bottom=0.21, left=0.12, right=0.88)

for col, (error_x, log) in enumerate(zip(all_error_x, log_files)):
    title = log["name"]
    is_bold = log.get("b", False)

    im = axs[0, col].imshow(error_x, origin="lower", cmap=cmap,
                            extent=[v_x_vals[0], v_x_vals[-1], omega_z_vals[0], omega_z_vals[-1]],
                            aspect='equal', vmin=vmin, vmax=vmax)
    axs[0, col].tick_params(labelbottom=False)
    axs[0, col].grid(alpha=0.2)
    if col != 0:
        axs[0, col].tick_params(labelleft=False)

    axs[0, col].set_title(
        title,
        fontname='DejaVu Serif',
        fontweight='bold' if is_bold else 'normal',
        color=cmap(0.08) if is_bold else 'black'
    )

    for spine in axs[0, col].spines.values():
        spine.set_linewidth(1.3)
    axs[0, col].tick_params(width=1.3, length=6)
    axs[0, col].spines['right'].set_visible(False)
    axs[0, col].spines['top'].set_visible(False)

for col, error_omega in enumerate(all_error_omega):
    im = axs[1, col].imshow(error_omega, origin="lower", cmap=cmap,
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
