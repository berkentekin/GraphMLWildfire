import argparse

from .splits import create_fire_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reproducible fire-level subsampling manifest"
    )
    parser.add_argument("data_dir", help="Directory containing processed HDF5 files")
    parser.add_argument("output_path", help="CSV manifest to create")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_fire_manifest(args.data_dir, args.output_path, args.fraction, args.seed)


if __name__ == "__main__":
    main()
