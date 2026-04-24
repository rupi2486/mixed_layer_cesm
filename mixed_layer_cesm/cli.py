import argparse
import numpy as np
from mixed_layer_cesm.calculate import compute_mld


def main():
    parser = argparse.ArgumentParser(description="Compute Mixed Layer Depth")

    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--time", type=str, required=True, help="Time (YYYY-MM-DD)")

    args = parser.parse_args()

    # Call your function
    z, rho, mld = compute_mld(args.lat, args.lon, args.time)

    print(f"MLD: {mld:.2f} meters")

    # Optional: save output
    np.savez(
        "mld_output.npz",
        depth=z,
        density=rho,
        mld=mld
    )


if __name__ == "__main__":
    main()