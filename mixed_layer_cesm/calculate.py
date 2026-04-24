import cftime
import numpy as np
import xarray as xr
import scipy.ndimage as nd
import gsw
from mixed_layer_cesm import open_cesm2le
from mixed_layer_cesm.core import _OCN_GRID_URL, _STORAGE_OPTIONS


def compute_mld(lat, lon, time):
    # Parse "YYYY-MM-DD" string into a cftime object so sel(method="nearest")
    # can compare it against the noleap calendar timestamps in the zarr store.
    year, month, day = map(int, time.split("-"))
    t = cftime.DatetimeNoLeap(year, month, day)

    # -----------------------------
    # Load datasets (time slice only)
    # -----------------------------
    ds_temp = open_cesm2le(
        "TEMP",
        component="ocn",
        scenario="historical",
        forcing="cmip6",
        members=0,
    ).sel(time=t, method="nearest").load()

    ds_salt = open_cesm2le(
        "SALT",
        component="ocn",
        scenario="historical",
        forcing="cmip6",
        members=0,
    ).sel(time=t, method="nearest").load()

    # -----------------------------
    # Align datasets
    # -----------------------------
    ds_temp, ds_salt = xr.align(ds_temp, ds_salt, join="inner")

    # Drop any extra dims (member_id, time) so the rest of the function
    # always works with a clean (z_t, nlat, nlon) array.
    # cftime string selection may return a size-1 time dim instead of a scalar.
    _spatial_dims = {'z_t', 'nlat', 'nlon'}
    for _dim in [d for d in ds_temp.dims if d not in _spatial_dims]:
        ds_temp = ds_temp.isel({_dim: 0})
        ds_salt = ds_salt.isel({_dim: 0})

    # -----------------------------
    # Get grid (TLAT/TLONG are 2-D on the curvilinear POP grid)
    # -----------------------------
    grid = xr.open_zarr(_OCN_GRID_URL, storage_options=_STORAGE_OPTIONS, consolidated=True)
    lat2d = grid["TLAT"].values   # (nlat, nlon)
    lon2d = grid["TLONG"].values  # (nlat, nlon), 0-360

    # -----------------------------
    # Handle longitude wrapping
    # -----------------------------
    lon_diff = np.abs(lon2d - lon)
    lon_diff = np.minimum(lon_diff, 360 - lon_diff)

    # -----------------------------
    # Mask invalid (land) points
    # -----------------------------
    surface_temp = ds_temp.isel(z_t=0).values  # (nlat, nlon)
    mask = np.isnan(surface_temp)

    # -----------------------------
    # Distance to all valid points
    # -----------------------------
    dist = (lat2d - lat)**2 + lon_diff**2
    dist[mask] = np.nan

    # -----------------------------
    # Find nearest valid ocean point
    # -----------------------------
    j, i = np.unravel_index(np.nanargmin(dist), dist.shape)

    # -----------------------------
    # Extract vertical profiles
    # -----------------------------
    da_temp = ds_temp.isel(nlat=j, nlon=i).squeeze()
    da_salt = ds_salt.isel(nlat=j, nlon=i).squeeze()

    # Safety check
    if da_temp.size == 0 or da_salt.size == 0:
        raise ValueError("Selected point has no valid ocean data")

    # -----------------------------
    # Depth (m)
    # -----------------------------
    z = da_temp["z_t"].values / 100.0
    z_m = da_temp["z_t"] / 100.0

    # -----------------------------
    # Pressure
    # -----------------------------
    lat_point = float(lat2d[j, i])

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
