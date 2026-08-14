import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.download_kuairec import (
    ARCHIVE_URL,
    SUBSET_FILES,
    download_archive,
    extract,
    main,
)


def make_archive(path: Path, names) -> Path:
    """Build a zip that mimics KuaiRec's nested layout ("KuaiRec 2.0/data/...")."""
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(f"KuaiRec 2.0/data/{name}", b"col_a,col_b\n1,2\n")
    return path


def test_archive_url_points_at_zenodo():
    # The previous per-file Aliyun URLs 404'd; the archive is the only real source.
    assert ARCHIVE_URL.startswith("https://zenodo.org/records/")
    assert ARCHIVE_URL.endswith("KuaiRec.zip")


def test_subsets_declare_expected_matrices():
    assert "small_matrix.csv" in SUBSET_FILES["small"]
    assert "big_matrix.csv" in SUBSET_FILES["big"]
    # The ranker needs side features in both subsets.
    for subset in SUBSET_FILES.values():
        assert "item_categories.csv" in subset
        assert "user_features.csv" in subset


def test_download_archive_calls_urlretrieve(tmp_path):
    dest = tmp_path / "KuaiRec.zip"

    def fake_urlretrieve(url, dst, reporthook=None):
        Path(dst).write_bytes(b"zip")

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve) as mock:
        download_archive(dest)

    assert mock.call_args[0][0] == ARCHIVE_URL
    assert dest.exists()


def test_download_archive_writes_via_part_file(tmp_path):
    """An interrupted transfer must not leave a file that looks complete."""
    dest = tmp_path / "KuaiRec.zip"
    seen = {}

    def fake_urlretrieve(url, dst, reporthook=None):
        seen["path"] = Path(dst)
        Path(dst).write_bytes(b"zip")

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        download_archive(dest)

    assert seen["path"].name.endswith(".part")
    assert dest.exists()


def test_download_archive_skips_existing(tmp_path):
    dest = tmp_path / "KuaiRec.zip"
    dest.write_bytes(b"already here")

    with patch("urllib.request.urlretrieve") as mock:
        download_archive(dest)

    mock.assert_not_called()


def test_reporthook_handles_unknown_total_size(tmp_path):
    dest = tmp_path / "KuaiRec.zip"

    def fake_urlretrieve(url, dst, reporthook=None):
        Path(dst).write_bytes(b"zip")

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve) as mock:
        download_archive(dest)

    reporthook = mock.call_args[1]["reporthook"]
    reporthook(1, 512, 1024)
    reporthook(0, 0, 0)      # total_size unknown — must not divide by zero


def test_extract_flattens_nested_paths(tmp_path):
    archive = make_archive(tmp_path / "a.zip", SUBSET_FILES["small"])
    out = tmp_path / "out"

    extract(archive, SUBSET_FILES["small"], out)

    for name in SUBSET_FILES["small"]:
        assert (out / name).exists(), f"{name} should be flattened into {out}"


def test_extract_skips_already_extracted(tmp_path):
    archive = make_archive(tmp_path / "a.zip", SUBSET_FILES["small"])
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "small_matrix.csv"
    existing.write_bytes(b"do not overwrite")

    extract(archive, SUBSET_FILES["small"], out)

    assert existing.read_bytes() == b"do not overwrite"


def test_extract_raises_when_member_missing(tmp_path):
    archive = make_archive(tmp_path / "a.zip", ["item_categories.csv"])
    out = tmp_path / "out"

    with pytest.raises(SystemExit) as exc:
        extract(archive, SUBSET_FILES["small"], out)

    assert "small_matrix.csv" in str(exc.value)


def test_main_skips_download_when_files_present(tmp_path, monkeypatch):
    """Re-running after a completed download must not re-fetch 432 MB."""
    monkeypatch.setattr(sys, "argv", ["prog", "--subset", "small"])
    monkeypatch.setattr("scripts.download_kuairec.PROJECT_ROOT", tmp_path)

    raw = tmp_path / "datastore" / "raw" / "kuairec"
    raw.mkdir(parents=True)
    for name in SUBSET_FILES["small"]:
        (raw / name).write_bytes(b"data")

    with patch("scripts.download_kuairec.download_archive") as mock_dl:
        main()

    mock_dl.assert_not_called()


def test_main_downloads_when_files_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--subset", "small"])
    monkeypatch.setattr("scripts.download_kuairec.PROJECT_ROOT", tmp_path)

    raw = tmp_path / "datastore" / "raw" / "kuairec"

    def fake_download(dest):
        make_archive(dest, SUBSET_FILES["small"])

    with patch("scripts.download_kuairec.download_archive", side_effect=fake_download) as mock_dl:
        main()

    mock_dl.assert_called_once()
    for name in SUBSET_FILES["small"]:
        assert (raw / name).exists()
    # Archive is removed unless --keep-archive is passed.
    assert not (raw / "KuaiRec.zip").exists()
