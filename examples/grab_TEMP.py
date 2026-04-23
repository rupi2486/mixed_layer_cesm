"""
Load TREFHT (near-surface air temperature) over a spatial box in Colorado
for the historical period, average across the box, and plot a time series.

Run:
    python examples/grab_trefht.py
"""

import matplotlib.pyplot as plt

from mixed_layer_cesm import open_cesm2le

# Open lazily — no data downloaded yet
da = open_cesm2le(
    "TEMP",
    component="ocn",
    scenario="historical",
    forcing="cmip6",
    time_slice=("1990-01", "2000-12"),
    lat=slice(20,25),      # Colorado-ish
    lon=slice(180,185),  # negative °W values; converted to 0–360 internally
    members=0,                  # first ensemble member only
)

print("Lazy DataArray before loading:")
print(da, "\n")

# Download the subset (small — one member, 11 years, ~handful of grid cells)
da = da.load()

# Spatial mean → time series, convert K → °C
ts = da.mean(["lat", "lon"]) - 273.15

fig, ax = plt.subplots(figsize=(10, 4))
ts.plot(ax=ax)
ax.set_title("CESM2-LE TREFHT — Colorado box (member 0, historical)")
ax.set_ylabel("Temperature (°C)")
ax.set_xlabel("Time")
plt.tight_layout()
plt.savefig("trefht_colorado.png", dpi=150)
print("Saved trefht_colorado.png")
plt.show()
