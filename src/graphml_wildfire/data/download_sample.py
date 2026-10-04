import argparse
from pathlib import Path
from zipfile import ZipFile

import requests
from remotezip import RemoteZip
from tqdm import tqdm

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
    print("Additional disk space is needed temporarily for the ZIP and extracted TIFFs.")

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download WildfireSpreadTS data"
    )
    parser.add_argument("--raw-dir", default="data/raw/wildfirespreadts")
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
        download_full_dataset(args.raw_dir)
        return

    if args.year is None:
        parser.error("--year is required unless --full-dataset is used")

    files = download_sample(args.raw_dir, args.year, args.fire)
    print(f"Downloaded data is ready under {Path(args.raw_dir) / str(args.year)}")
    print(f"Available GeoTIFF files: {len(files)}")


if __name__ == "__main__":
    main()
