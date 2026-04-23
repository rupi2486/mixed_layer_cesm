import argparse
from mixed_layer_cesm.calculate import compute_mld
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run Mixed Layer Depth calculation"
    )

    parser.add_argument("--lat", type=float, required=True,
                        help="Latitude (single value)")

    parser.add_argument("--lon", type=float, required=True,
                        help="Longitude (single value)")

    parser.add_argument("--time", type=str, required=True,
                        help="Date (YYYY-MM-DD)")

    parser.add_argument("-o", "--output",
                        default="data_cache/mld_results.npz",
                        help="Output file path")

    args = parser.parse_args()

    # run core computation (NOW returns tuple)
    z, rho_smooth, mld_value = compute_mld(
        args.lat,
        args.lon,
        args.time
    )

    # save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    np.savez(
        args.output,
        z=z,
        rho_smooth=rho_smooth,
        mld_value=mld_value,
        lat=args.lat,
        lon=args.lon,
        time=args.time
    )

    print(f"Saved MLD results → {args.output}")
    print(f"MLD = {mld_value:.2f} m")


if __name__ == "__main__":
    main()