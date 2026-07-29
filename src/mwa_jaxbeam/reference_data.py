"""Download and locate upstream MWA beam-model reference data."""

from __future__ import annotations

from pathlib import Path

import wget


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent

DEFAULT_REFERENCE_DIR = REPO_ROOT / "data"

JMATRIX_URL = (
    "https://raw.githubusercontent.com/"
    "MWATelescope/mwa_pb/master/mwa_pb/data/Jmatrix.fits"
)

ZMATRIX_URL = (
    "https://raw.githubusercontent.com/"
    "MWATelescope/mwa_pb/master/mwa_pb/data/ZMatrix.fits"
)

LNA_IMPEDANCE_URL = (
    "https://raw.githubusercontent.com/"
    "RadioAstronomySoftwareGroup/pyuvdata/main/"
    "src/pyuvdata/data/mwa_config_data/mwa_lna_impedance.txt"
)


def download_if_missing(path: Path, url: str) -> None:
    """Download a file if it is not already present."""
    path = Path(path)

    if path.is_file():
        print(f"Found {path.name}")
        return

    if path.exists():
        raise FileExistsError(
            f"{path} exists but is not a regular file."
        )

    print(f"Downloading {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    wget.download(url, out=str(path))
    print()


def ensure_reference_data(
    *,
    jmatrix_path: Path,
    zmatrix_path: Path,
    lna_impedance_path: Path,
) -> None:
    """Ensure that the upstream AEE reference files are available."""
    download_if_missing(
        jmatrix_path,
        JMATRIX_URL,
    )
    download_if_missing(
        zmatrix_path,
        ZMATRIX_URL,
    )
    download_if_missing(
        lna_impedance_path,
        LNA_IMPEDANCE_URL,
    )
