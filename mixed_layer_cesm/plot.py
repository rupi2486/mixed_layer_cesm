import matplotlib.pyplot as plt

from calculate import (
    create_profiles,
    density_profile,
    density_gradient,
    find_mld
)

# -----------------------------
# Run the pipeline
# -----------------------------
z, T, S = create_profiles()

rho = density_profile(T, S, z)
drho_dz = density_gradient(z, rho)

mld = find_mld(z, drho_dz, threshold=0.01)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(5,6))

plt.plot(rho, z, label="Density")

# mark mixed layer depth
if mld is not None:
    plt.axhline(mld, linestyle='--', label=f"MLD ≈ {mld:.1f} m")

plt.gca().invert_yaxis()
plt.xlabel("Density (kg/m³)")
plt.ylabel("Depth (m)")
plt.title("Density Profile and Mixed Layer Depth")
plt.legend()

plt.show()