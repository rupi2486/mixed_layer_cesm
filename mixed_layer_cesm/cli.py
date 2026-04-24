import argparse
import numpy as np
from mixed_layer_cesm.calculate import compute_mld


def main():
    parser = argparse.ArgumentParser(description="Compute Mixed Layer Depth")

    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--time", type=str, required=True, help="Target date (YYYY-MM-DD); nearest monthly timestep is selected")
    parser.add_argument("--plot", action="store_true", help="Show density profile plot with MLD marked")

    args = parser.parse_args()

    z, rho, mld = compute_mld(args.lat, args.lon, args.time)

    print(f"MLD: {mld:.2f} meters")

    np.savez("mld_output.npz", depth=z, density=rho, mld=mld)

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(rho, z, label="Density")
        if not np.isnan(mld):
            plt.axhline(mld, color="red", linestyle="--", label="MLD")
            plt.text(
                0.98, 0.95,
                f"MLD = {mld:.1f} m",
                transform=plt.gca().transAxes,
                ha="right", va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
        plt.gca().invert_yaxis()
        plt.xlabel("Density (kg/m³)")
        plt.ylabel("Depth (m)")
        plt.title(f"lat={args.lat}, lon={args.lon}, time={args.time}")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
