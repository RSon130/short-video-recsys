"""
Download the KuaiRec dataset into datastore/raw/kuairec/.

KuaiRec is distributed as a single ~432 MB zip archive, not as individual CSVs.
(An earlier version of this script fetched per-file URLs from an Aliyun OSS
bucket; those paths return 404 and never existed.) The canonical source is the
Zenodo record linked from the official site, https://kuairec.com/.

Usage:
    python scripts/download_kuairec.py                 # small subset (Tier 1)
    python scripts/download_kuairec.py --subset big    # adds big_matrix.csv
    python scripts/download_kuairec.py --keep-archive  # don't delete the zip
"""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD = "18164998"
ARCHIVE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/KuaiRec.zip"

PROJECT_ROOT = Path(__file__).parent.parent

# Files needed per subset. Names are matched against the archive's basenames,
# so this survives changes to the directory layout inside the zip.
SUBSET_FILES = {
    "small": [
        "small_matrix.csv",
        "item_categories.csv",
        "user_features.csv",
    ],
    "big": [
        "big_matrix.csv",
        "small_matrix.csv",
        "item_categories.csv",
        "item_daily_features.csv",
        "user_features.csv",
    ],
}


def download_archive(dest: Path) -> None:
    """Fetch KuaiRec.zip, printing progress. Re-runs skip a completed download."""
    if dest.exists():
        print(f"Archive already present: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return

    # Only emit on a percentage change. urlretrieve calls the hook once per 8 KB
    # block, which is ~53,000 calls for this archive — enough to bury real output
    # in any captured log.
    last_pct = -1

    def reporthook(block_num, block_size, total_size):
        nonlocal last_pct
        if total_size <= 0:
            return
        pct = min(int(block_num * block_size * 100 / total_size), 100)
        if pct != last_pct:
            last_pct = pct
            got = block_num * block_size / 1e6
            print(f"\rKuaiRec.zip: {pct}%  ({got:.0f}/{total_size / 1e6:.0f} MB)",
                  end="", flush=True)

    # Download to a temp name so an interrupted transfer is never mistaken for
    # a complete archive on the next run.
    tmp = dest.with_suffix(".zip.part")
    print(f"Downloading {ARCHIVE_URL}")
    urllib.request.urlretrieve(ARCHIVE_URL, tmp, reporthook=reporthook)
    print()
    tmp.rename(dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


def extract(archive: Path, wanted: list, output_dir: Path) -> None:
    """
    Extract the requested CSVs from the archive, flattening them into output_dir.

    The zip nests files under a versioned directory (e.g. "KuaiRec 2.0/data/").
    Flattening keeps the loader's config paths independent of that layout.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    remaining = set(wanted)

    with zipfile.ZipFile(archive) as zf:
        members = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
        for name in wanted:
            dest = output_dir / name
            if dest.exists():
                print(f"  {name} — already extracted")
                remaining.discard(name)
                continue
            member = members.get(name)
            if member is None:
                continue
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            print(f"  {name} — {dest.stat().st_size / 1e6:.1f} MB")
            remaining.discard(name)

    if remaining:
        raise SystemExit(
            f"Not found in archive: {sorted(remaining)}\n"
            f"Archive contents may have changed — inspect {archive} and update "
            f"SUBSET_FILES."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["small", "big"], default="small")
    parser.add_argument("--keep-archive", action="store_true",
                        help="keep KuaiRec.zip after extraction (default: delete)")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "datastore" / "raw" / "kuairec"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "KuaiRec.zip"

    wanted = SUBSET_FILES[args.subset]
    if all((output_dir / f).exists() for f in wanted):
        print(f"All '{args.subset}' files already present in {output_dir}")
        return

    download_archive(archive)
    print(f"Extracting '{args.subset}' subset to {output_dir}")
    extract(archive, wanted, output_dir)

    if not args.keep_archive:
        archive.unlink()
        print("Removed KuaiRec.zip (pass --keep-archive to retain it)")

    print("Done.")


if __name__ == "__main__":
    main()
