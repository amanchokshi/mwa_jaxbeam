"""Download and locate upstream MWA beam-model reference data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copyfileobj
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent.parent

DEFAULT_REFERENCE_DIR = REPO_ROOT / "data"

DEFAULT_RUNTIME_ARCHIVE = (
    PACKAGE_DIR
    / "data"
    / "aee_137mhz.npz"
)


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


@dataclass(frozen=True, slots=True)
class ReferenceDataPaths:
    jmatrix: Path
    zmatrix: Path
    lna_impedance: Path


def default_reference_paths(
    data_dir: Path = DEFAULT_REFERENCE_DIR,
) -> ReferenceDataPaths:
    """Return the default local paths for the reference files."""
    data_dir = Path(data_dir)

    return ReferenceDataPaths(
        jmatrix=data_dir / "Jmatrix.fits",
        zmatrix=data_dir / "ZMatrix.fits",
        lna_impedance=data_dir / "mwa_lna_impedance.txt",
    )


def download_if_missing(
    path: Path,
    url: str,
    *,
    timeout_seconds: float = 60.0,
) -> None:
    """
    Download a file atomically if it is not already present.

    The download is first written to a temporary ``.part`` file. The
    destination is replaced only after a non-empty download completes.
    """
    path = Path(path)

    if path.is_file():
        print(f"Found {path.name}")
        return

    if path.exists():
        raise FileExistsError(f"{path} exists but is not a regular file.")

    print(f"Downloading {path.name} from:")
    print(f"  {url}")

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(f"{path.suffix}.part")
    temporary_path.unlink(missing_ok=True)

    try:
        with (
            urlopen(url, timeout=timeout_seconds) as response,
            temporary_path.open("wb") as output_file,
        ):
            copyfileobj(response, output_file)
    except (HTTPError, URLError, TimeoutError, OSError):
        temporary_path.unlink(missing_ok=True)
        raise

    if not temporary_path.is_file():
        raise RuntimeError(f"Download did not produce a regular file: {url}")

    if temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is empty: {url}")

    temporary_path.replace(path)
    print(f"Saved {path}")


def ensure_reference_data(
    *,
    jmatrix_path: Path | None = None,
    zmatrix_path: Path | None = None,
    lna_impedance_path: Path | None = None,
    data_dir: Path = DEFAULT_REFERENCE_DIR,
) -> ReferenceDataPaths:
    """
    Ensure that the upstream AEE reference files are available.

    Explicit paths may be supplied individually. Any path not supplied uses
    the corresponding default path under ``data_dir``. Missing files are
    downloaded to their resolved locations.

    Parameters
    ----------
    jmatrix_path
        Local path to ``Jmatrix.fits``.
    zmatrix_path
        Local path to ``ZMatrix.fits``.
    lna_impedance_path
        Local path to ``mwa_lna_impedance.txt``.
    data_dir
        Directory used for any paths that are not supplied explicitly.

    Returns
    -------
    ReferenceDataPaths
        Resolved paths to all three reference files.
    """
    defaults = default_reference_paths(data_dir)

    paths = ReferenceDataPaths(
        jmatrix=(Path(jmatrix_path) if jmatrix_path is not None else defaults.jmatrix),
        zmatrix=(Path(zmatrix_path) if zmatrix_path is not None else defaults.zmatrix),
        lna_impedance=(
            Path(lna_impedance_path)
            if lna_impedance_path is not None
            else defaults.lna_impedance
        ),
    )

    download_if_missing(paths.jmatrix, JMATRIX_URL)
    download_if_missing(paths.zmatrix, ZMATRIX_URL)
    download_if_missing(
        paths.lna_impedance,
        LNA_IMPEDANCE_URL,
    )

    return paths
