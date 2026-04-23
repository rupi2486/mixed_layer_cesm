import numpy as np
import xarray as xr
import scipy.ndimage as nd
import gsw
import os
from mixed_layer_cesm import open_cesm2le


def compute_mld(lat, lon, time):

    # -----------------------------
    # Load temperature
    # -----------------------------
    ds_temp = open_cesm2le(
        "TEMP",
        component="ocn",
        scenario="historical",
        forcing="cmip6",
        members=0,
    ).sel(time=time).load()

    # -----------------------------
    # Load salinity
    # -----------------------------
    ds_salt = open_cesm2le(
        "SALT",
        component="ocn",
        scenario="historical",
        forcing="cmip6",
        members=0,
    ).sel(time=time).load()

    lat2d = ds_temp["nlat"]
    lon2d = ds_temp["nlon"]

    dist = (lat2d - lat)**2 + (lon2d - lon)**2
    j, i = np.unravel_index(np.argmin(dist.values), dist.shape)

    da_temp = ds_temp.isel(nlat=j, nlon=i)
    da_salt = ds_salt.isel(nlat=j, nlon=i)
    
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
        lat,
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
    # Profile (already single point)
    # -----------------------------
    rho_vals = rho.squeeze().values

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

    return z, rho_smooth, mld_value