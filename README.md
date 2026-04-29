# mixed_layer_cesm

A lightweight Python package for accessing the [CESM2 Large Ensemble](https://www.cesm.ucar.edu/community-projects/lens2) (CESM2-LE) hosted on the [NCAR AWS S3 archive](https://registry.opendata.aws/ncar-cesm2-lens/). No account or credentials required — the bucket is publicly accessible. To determine the density profile and mixed layer depth of a specific latitude, longitude, and date.
---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10. Core dependencies (`xarray`, `zarr`, `s3fs`, `numpy`) are installed automatically. For plotting:

```bash
pip install -e .
```

---

## Quick start
run mld-cli --lat ## --lon ### --time ####-##-## --plot (replace the # with your lat lon and date) in terminal. For latitude, North is positive and South is negative. For longitude, East is positive and West is negative. If you don't need the plot but just the mld, do mld-cli --lat ## --lon ### --time ####-##-##