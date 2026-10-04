import argparse
from pathlib import Path
from zipfile import ZipFile

import requests
from remotezip import RemoteZip
from tqdm import tqdm

from .convert_hdf5 import convert_dataset, convert_fire


ARCHIVE_URL = "https://zenodo.org/api/records/8006177/files/WildfireSpreadTS.zip/content"


def select_files(
    names: list[str],
    year: int,
    fire_name: str | None = None,
) -> list[str]:
    """Select TIFF files belonging to one fire or a complete year."""

    prefix = f"{year}/{fire_name}/" if fire_name else f"{year}/"
    return sorted(
        name for name in names if name.startswith(prefix) and name.endswith(".tif")
    )


def download_sample(
    output_dir: str | Path,
    year: int,
    fire_name: str | None = None,
) -> list[Path]:
    """Download one fire or year without downloading the complete ZIP archive."""

    output_dir = Path(output_dir)
    with RemoteZip(ARCHIVE_URL) as archive:
        files = select_files(archive.namelist(), year, fire_name)
        if not files:
            selection = f"{year}/{fire_name}" if fire_name else str(year)
            raise ValueError(f"No files found for {selection}")

        info_by_name = {info.filename: info for info in archive.infolist()}
        size_mb = sum(info_by_name[name].compress_size for name in files) / 1024**2
        missing = [name for name in files if not (output_dir / name).exists()]

        print(f"Selected {len(files)} files ({size_mb:.1f} MiB compressed)")
        if missing:
            for name in tqdm(missing, desc="Downloading", unit="file"):
                archive.extract(name, output_dir)

    if not missing:
        print("All selected files already exist")
    return [output_dir / name for name in files]


def download_full_dataset(output_dir: str | Path) -> None:
    """Download and extract the complete WildfireSpreadTS archive."""

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir.parent / "WildfireSpreadTS.zip"
    partial_path = archive_path.with_suffix(".zip.part")

    print("WARNING: The full dataset download is about 50 GB and may take many hours.")
    print("Additional disk space is needed for the ZIP, extracted TIFFs, and HDF5 files.")

    if not archive_path.exists():
        with requests.get(ARCHIVE_URL, stream=True, timeout=(30, None)) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with partial_path.open("wb") as output, tqdm(
                total=total_size or None,
                desc="Downloading archive",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        progress.update(len(chunk))
        partial_path.replace(archive_path)
    else:
        print(f"Using existing archive: {archive_path}")

    with ZipFile(archive_path) as archive:
        files = [info for info in archive.infolist() if info.filename.endswith(".tif")]
        for info in tqdm(files, desc="Extracting", unit="file"):
            archive.extract(info, output_dir)

    archive_path.unlink()
    print(f"Full dataset extracted to {output_dir}")


def prepare_fire(
    raw_dir: str | Path,
    processed_dir: str | Path,
    year: int,
    fire_name: str,
) -> Path:
    """Download one fire and convert it to HDF5."""

    download_sample(raw_dir, year, fire_name)
    fire_dir = Path(raw_dir) / str(year) / fire_name
    output_path = Path(processed_dir) / str(year) / f"{fire_name}.hdf5"
    with tqdm(total=1, desc="Converting fires", unit="fire") as progress:
        progress.set_postfix_str(f"{year}/{fire_name}")
        convert_fire(fire_dir, output_path, overwrite=True)
        progress.update()
    print(f"HDF5 file ready: {output_path}")
    return output_path


def prepare_year(
    raw_dir: str | Path,
    processed_dir: str | Path,
    year: int,
) -> list[Path]:
    """Download all fires from one year and convert them to HDF5."""

    download_sample(raw_dir, year)
    converted = convert_dataset(
        raw_dir,
        processed_dir,
        years=[year],
        overwrite=True,
    )
    print(f"Created {len(converted)} HDF5 files for {year}")
    return converted


def prepare_full_dataset(raw_dir: str | Path, processed_dir: str | Path) -> list[Path]:
    """Download the full dataset and convert every fire to HDF5."""

    download_full_dataset(raw_dir)
    converted = convert_dataset(raw_dir, processed_dir, overwrite=True)
    print(f"Created {len(converted)} HDF5 files")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download WildfireSpreadTS data"
    )
    parser.add_argument("--raw-dir", default="data/raw/wildfirespreadts")
    parser.add_argument("--processed-dir", default="data/processed/wildfirespreadts")
    parser.add_argument("--year", type=int, help="Year to download")
    parser.add_argument(
        "--fire",
        help="Fire ID to download; omit to download every fire in the year",
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Download and extract the complete approximately 50 GB dataset",
    )
    args = parser.parse_args()

    if args.full_dataset:
        if args.year is not None or args.fire is not None:
            parser.error("--full-dataset cannot be combined with --year or --fire")
        prepare_full_dataset(args.raw_dir, args.processed_dir)
        return

    if args.year is None:
        parser.error("--year is required unless --full-dataset is used")

    if args.fire is None:
        prepare_year(args.raw_dir, args.processed_dir, args.year)
        return

    output_path = prepare_fire(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        year=args.year,
        fire_name=args.fire,
    )
    print(f"Ready to load: {output_path}")


if __name__ == "__main__":
    main()
