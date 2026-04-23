import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load saved results
# -----------------------------
data = np.load("data_cache/mld_results.npz")

z = data["z"]
rho_smooth = data["rho_smooth"]
mld_value = data["mld_value"]

# -----------------------------
# Plot
# -----------------------------
plt.figure()

plt.plot(rho_smooth, z, label="Density")

if not np.isnan(mld_value):
    plt.axhline(mld_value, color="red", linestyle="--", label="MLD")

    plt.text(
        0.98, 0.95,
        f"MLD = {mld_value:.1f} m",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

plt.gca().invert_yaxis()
plt.xlabel("Density (kg/m³)")
plt.ylabel("Depth (m)")
plt.legend()
plt.show()