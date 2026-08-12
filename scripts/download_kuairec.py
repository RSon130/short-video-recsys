"""
Download KuaiRec dataset files to data/raw/kuairec/.

Usage:
    python scripts/download_kuairec.py --subset small
    python scripts/download_kuairec.py --subset big
"""
import argparse
import os
import urllib.request
from pathlib import Path

BASE_URL = "https://kuairec.oss-cn-beijing.aliyuncs.com/"

SUBSET_FILES = {
    "small": ["small_matrix.csv", "item_categories.csv", "user_features.csv"],
    "big": ["big_matrix.csv", "item_categories.csv", "item_daily_features.csv", "user_features.csv"],
}


def download_file(url: str, dest: Path) -> None:
    """
    Download a single file from `url` to `dest`, with a progress indicator.

    Skips the download silently if the destination file already exists,
    making the script safe to re-run without re-downloading completed files.

    Uses urllib.request.urlretrieve (stdlib only — no extra dependencies)
    with a reporthook that prints percentage progress on a single overwritten
    line using carriage return.

    Args:
        url:  Full URL of the file to download.
        dest: Local destination path (parent directory must already exist).
    """
    if dest.exists():
        print(f"Skipping {dest.name} — already exists")
        return

    filename = dest.name

    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(int(block_num * block_size * 100 / total_size), 100)
            print(f"\r{filename}: {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Saved to {dest} ({size_mb:.2f} MB)")


def main() -> None:
    """
    CLI entry point.

    Parses --subset (small | big), resolves the output directory relative to
    the project root, and calls download_file() for each file in the subset.
    The 'small' subset (small_matrix.csv + supporting files) is sufficient
    for all Tier 1 development; 'big' downloads the full 12.5M interaction
    matrix for Tier 2 scale-up.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["small", "big"], default="small")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "raw" / "kuairec"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in SUBSET_FILES[args.subset]:
        dest = output_dir / filename
        if dest.exists():
            print(f"Skipping {filename} — already exists")
            continue
        url = BASE_URL + filename
        download_file(url, dest)


if __name__ == "__main__":
    main()
