import numpy as np
import xarray as xr
import scipy.ndimage as nd
import gsw
import os
from mixed_layer_cesm import open_cesm2le

# -----------------------------
# Load temperature
# -----------------------------
da_temp = open_cesm2le(
    "TEMP",
    component="ocn",
    scenario="historical",
    forcing="cmip6",
    lat=slice(20, 25),
    lon=slice(180, 185),
    members=0,
).sel(time="1990-02").load()

# -----------------------------
# Load salinity
# -----------------------------
da_salt = open_cesm2le(
    "SALT",
    component="ocn",
    scenario="historical",
    forcing="cmip6",
    lat=slice(20, 25),
    lon=slice(180, 185),
    members=0,
).sel(time="1990-02").load()

# -----------------------------
# Depth (m)
# -----------------------------
z = da_temp["z_t"].values / 100.0
z = np.squeeze(z)

z_m = da_temp["z_t"] / 100.0

# -----------------------------
# Pressure
# -----------------------------
p = xr.apply_ufunc(
    gsw.p_from_z,
    -z_m,
    22.5,
    vectorize=True
)

# -----------------------------
# Conservative temperature
# -----------------------------
CT = xr.apply_ufunc(
    gsw.CT_from_t,
    da_salt,
    da_temp,
    p,
    vectorize=True
)

# -----------------------------
# Density
# -----------------------------
rho = xr.apply_ufunc(
    gsw.rho,
    da_salt,
    CT,
    p,
    vectorize=True
)

# -----------------------------
# Horizontal mean profile
# -----------------------------
rho_prof = rho.mean(dim=["nlat", "nlon"]).squeeze()
rho_vals = rho_prof.values

# -----------------------------
# sort + smooth
# -----------------------------
sort_idx = np.argsort(z)
z = z[sort_idx]
rho_vals = rho_vals[sort_idx]

rho_smooth = nd.gaussian_filter1d(rho_vals, sigma=2)

# -----------------------------
# gradient + MLD
# -----------------------------
drho_dz = np.gradient(rho_smooth, z)

threshold = 0.01
idx = np.where(np.abs(drho_dz) > threshold)[0]

mld_value = z[idx[0]] if len(idx) > 0 else np.nan

# -----------------------------
# SAVE RESULTS
# -----------------------------
os.makedirs("data_cache", exist_ok=True)
np.savez(
    "data_cache/mld_results.npz",
    z=z,
    rho_smooth=rho_smooth,
    mld_value=mld_value
)

print("Saved MLD results.")