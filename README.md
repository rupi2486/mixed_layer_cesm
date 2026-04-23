# grab-cesm

A lightweight Python package for accessing the [CESM2 Large Ensemble](https://www.cesm.ucar.edu/community-projects/lens2) (CESM2-LE) hosted on the [NCAR AWS S3 archive](https://registry.opendata.aws/ncar-cesm2-lens/). No account or credentials required — the bucket is publicly accessible.

Data are opened **lazily** via [Zarr](https://zarr.dev/) and [xarray](https://docs.xarray.dev/), so you only download the chunks you actually use.

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10. Core dependencies (`xarray`, `zarr`, `s3fs`, `numpy`) are installed automatically. For plotting:

```bash
pip install -e ".[plot]"
```

---

## Quick start

### Discover available variables

```python
from grab_cesm import list_variables

# All atmosphere variables, historical, CMIP6 forcing
list_variables("atm", "historical", "cmip6")

# All ocean variables, SSP3-7.0 scenario
list_variables("ocn", "ssp370", "cmip6")
```

Valid options:

| Argument    | Values                     |
|-------------|----------------------------|
| `component` | `"atm"`, `"ocn"`           |
| `scenario`  | `"historical"`, `"ssp370"` |
| `forcing`   | `"cmip6"`, `"smbb"`        |

### Load a variable

```python
from grab_cesm import open_cesm2le

da = open_cesm2le(
    "TREFHT",                       # variable name
    component="atm",
    scenario="historical",
    forcing="cmip6",
    time_slice=("1990-01", "2000-12"),
    lat=40.0,                       # scalar → nearest grid point
    lon=-105.0,                     # negative °W fine; converted to 0–360 internally
    members=0,                      # 0-based index; None = all members
)

da.load().plot()
```

`open_cesm2le` returns a lazy `xr.DataArray`. Nothing is downloaded until you call `.load()` or `.compute()` — subset aggressively before loading.

---

## Spatial selection

### `lat` / `lon` argument forms

| Form | Behaviour |
|------|-----------|
| `scalar` (e.g. `40.0`) | Nearest grid point |
| `tuple` (e.g. `(37.0, 41.0)`) | Bounding range |
| `slice` (e.g. `slice(37.0, 41.0)`) | Bounding range (same as tuple) |
| `None` | No spatial subsetting |

### Atmosphere vs Ocean grids

**Atmosphere** (`component="atm"`) uses a regular lat/lon grid. All selection forms work directly.

**Ocean** (`component="ocn"`) uses the curvilinear [POP](https://pop-model.readthedocs.io/) grid, where latitude and longitude are stored as 2-D arrays (`TLAT`, `TLONG`). A scalar `lat`/`lon` finds the nearest grid cell; a tuple takes a conservative bounding-box slice over the `nlat`/`nlon` index dimensions.

---

## Ensemble members

The `member_id` dimension holds all ensemble members. Pass a 0-based integer or list of integers to select a subset:

```python
# Single member
da = open_cesm2le("TREFHT", members=0)

# First 10 members
da = open_cesm2le("TREFHT", members=list(range(10)))

# All members (default)
da = open_cesm2le("TREFHT")
```

---

## S3 path structure

Each variable lives in its own Zarr store:

```
s3://ncar-cesm2-lens/{component}/monthly/cesm2LE-{scenario}-{forcing}-{variable}.zarr
```

---

## Examples

```bash
# Print every available variable grouped by component/scenario/forcing
python examples/list_all_variables.py

# Load TREFHT over Colorado, plot a time series, save a PNG
python examples/grab_trefht.py
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `xarray` | Multi-dimensional arrays and lazy I/O |
| `zarr` | Zarr store backend |
| `s3fs` | Anonymous S3 access |
| `numpy` | Array operations |
| `matplotlib` *(optional)* | Plotting |
