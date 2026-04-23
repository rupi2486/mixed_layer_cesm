import argparse
from .core import open_geopotential, geostrophic_wind, load   # your imports

def main():
    parser = argparse.ArgumentParser(
        description="Compute mixed layer depth from temperature and salinity"
    )
    parser.add_argument("--level",  type=int,   default=500,
                        help="Pressure level in hPa (default: 500)")
    parser.add_argument("--lat",    type=float, nargs=2, default=[20.0, 60.0],
                        metavar=("SOUTH", "NORTH"))
    parser.add_argument("--lon",    type=float, nargs=2, default=[-135.0, -60.0],
                        metavar=("WEST", "EAST"))
    parser.add_argument("--time",   type=str,   nargs=2, required=True,
                        metavar=("START", "END"))
    parser.add_argument("-o", "--output", default="geowind_out.nc",
                        help="Output NetCDF filename")
    args = parser.parse_args()

    phi = open_geopotential(
        time_slice=tuple(args.time),
        level=args.level,
        lat=tuple(args.lat),
        lon=tuple(args.lon),
    )
    phi  = load(phi)
    ug, vg = geostrophic_wind(phi)
    ug.to_netcdf(args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()