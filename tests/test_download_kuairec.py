import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


def test_url_construction():
    from scripts.download_kuairec import BASE_URL, SUBSET_FILES

    assert BASE_URL == "https://kuairec.oss-cn-beijing.aliyuncs.com/"
    assert "small" in SUBSET_FILES
    assert "big" in SUBSET_FILES
    assert "small_matrix.csv" in SUBSET_FILES["small"]
    assert "big_matrix.csv" in SUBSET_FILES["big"]


def test_download_file_calls_urlretrieve(tmp_path):
    dest = tmp_path / "test.csv"

    def fake_urlretrieve(url, dst, reporthook=None):
        Path(dst).write_bytes(b"x")

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve) as mock_urlretrieve:
        from scripts.download_kuairec import download_file
        download_file("https://example.com/test.csv", dest)

    mock_urlretrieve.assert_called_once()
    assert mock_urlretrieve.call_args[0][0] == "https://example.com/test.csv"
    assert mock_urlretrieve.call_args[0][1] == dest


def test_reporthook_does_not_raise(tmp_path):
    dest = tmp_path / "f.csv"

    def fake_urlretrieve(url, dst, reporthook=None):
        Path(dst).write_bytes(b"x")

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve) as mock_urlretrieve:
        from scripts.download_kuairec import download_file
        download_file("https://x.com/f.csv", dest)

    reporthook = mock_urlretrieve.call_args[0][2]
    reporthook(1, 512, 1024)
    reporthook(0, 0, 0)


def test_download_file_skips_existing(tmp_path):
    dest = tmp_path / "existing.csv"
    dest.write_bytes(b"data")

    with patch("urllib.request.urlretrieve") as mock_urlretrieve:
        from scripts.download_kuairec import download_file
        download_file("https://x.com/existing.csv", dest)

    mock_urlretrieve.assert_not_called()


def test_main_calls_download_for_small(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--subset", "small"])
    with patch("scripts.download_kuairec.download_file") as mock_dl:
        from scripts.download_kuairec import main, SUBSET_FILES
        main()
    assert mock_dl.call_count == len(SUBSET_FILES["small"])
