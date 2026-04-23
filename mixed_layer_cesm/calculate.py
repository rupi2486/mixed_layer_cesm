import numpy as np
import xarray as xr
import scipy.ndimage as nd
import gsw
from mixed_layer_cesm import open_cesm2le


def compute_mld(lat, lon, time):

    # -----------------------------
    # Load datasets (time slice only)
    # -----------------------------
    ds_temp = open_cesm2le(
        "TEMP",
        component="ocn",
        scenario="historical",
        forcing="cmip6",
        members=0,
    ).sel(time=time).load()

    ds_salt = open_cesm2le(
        "SALT",
        component="ocn",
        scenario="historical",
        forcing="cmip6",
        members=0,
    ).sel(time=time).load()

    # -----------------------------
    # Align datasets
    # -----------------------------
    ds_temp, ds_salt = xr.align(ds_temp, ds_salt, join="inner")

    # -----------------------------
    # Get grid
    # -----------------------------
    lat2d = ds_temp["lat"]
    lon2d = ds_temp["lon"]

    # -----------------------------
    # Handle longitude wrapping
    # -----------------------------
    lon_model = lon2d.values
    lon_input = lon

    lon_diff = np.abs(lon_model - lon_input)
    lon_diff = np.minimum(lon_diff, 360 - lon_diff)

    # -----------------------------
    # Mask invalid (land) points
    # -----------------------------
    surface_temp = ds_temp.isel(z_t=0)
    mask = np.isnan(surface_temp)

    # -----------------------------
    # Distance to all valid points
    # -----------------------------
    dist = (lat2d - lat)**2 + lon_diff**2
    dist = dist.where(~mask)

    # -----------------------------
    # Find nearest valid ocean point
    # -----------------------------
    j, i = np.unravel_index(np.nanargmin(dist.values), dist.shape)

    # -----------------------------
    # Extract vertical profiles
    # -----------------------------
    da_temp = ds_temp.isel(nlat=j, nlon=i)
    da_salt = ds_salt.isel(nlat=j, nlon=i)

    # Safety check
    if da_temp.size == 0 or da_salt.size == 0:
        raise ValueError("Selected point has no valid ocean data")

    # -----------------------------
    # Depth (m)
    # -----------------------------
    z = (da_temp["z_t"].values / 100.0)
    z_m = da_temp["z_t"] / 100.0

    # -----------------------------
    # Pressure
    # -----------------------------
    lat_point = float(lat2d.values[j, i])

    p = xr.apply_ufunc(
        gsw.p_from_z,
        -z_m,
        lat_point,
    )

    # -----------------------------
    # Conservative temperature
    # -----------------------------
    CT = xr.apply_ufunc(
        gsw.CT_from_t,
        da_salt,
        da_temp,
        p,
    )

    # -----------------------------
    # Density
    # -----------------------------
    rho = xr.apply_ufunc(
        gsw.rho,
        da_salt,
        CT,
        p,
    )

    rho_vals = rho.values.squeeze()

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