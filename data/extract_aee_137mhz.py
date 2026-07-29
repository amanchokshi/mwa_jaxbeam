"""
Extract and preprocess the MWA Average Embedded Element model at 137 MHz.

The script:

1. Downloads the source files if they are missing.
2. Finds the J- and Z-matrix frequencies bracketing 137 MHz.
3. Interpolates the complex J and Z matrices to exactly 137 MHz.
4. Interpolates the LNA impedance to exactly 137 MHz.
5. Converts the source azimuth convention to:

       az = 0 at North, increasing eastward.

6. Saves a compact NumPy archive for the JAX runtime model.

Run from the repository root with:

    uv run python data/extract_aee_137mhz.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copyfileobj
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray


TARGET_FREQUENCY_HZ = 137e6

MWA_DIPOLE_SPACING_M = 1.1
MWA_NFEED = 2
MWA_NDIPOLE = 16
MWA_NPORT = MWA_NFEED * MWA_NDIPOLE

DEFAULT_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = (
    DEFAULT_DATA_DIR.parent / "src" / "mwa_jaxbeam" / "data" / "aee_137mhz.npz"
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


def download_if_missing(path: Path, url: str) -> None:
    """Download a file if it does not already exist."""
    if path.is_file():
        print(f"Found {path.name}")
        return

    if path.exists():
        raise FileExistsError(f"{path} exists but is not a regular file.")

    print(f"Downloading {path.name} from:")
    print(f"  {url}")

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".part")

    try:
        with (
            urlopen(url) as response,
            temporary_path.open("wb") as output_file,
        ):
            copyfileobj(response, output_file)
    except (HTTPError, URLError, OSError):
        temporary_path.unlink(missing_ok=True)
        raise

    if temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is empty: {url}")

    temporary_path.replace(path)
    print(f"Saved {path}")


def read_frequencies_hz(path: Path) -> NDArray[np.float64]:
    """Read the frequency associated with each FITS HDU."""
    frequencies: list[float] = []

    with fits.open(path, memmap=True) as hdul:
        for hdu_index, hdu in enumerate(hdul):
            if "FREQ" not in hdu.header:
                raise KeyError(f"HDU {hdu_index} in {path} has no FREQ header.")

            frequencies.append(float(hdu.header["FREQ"]))

    frequencies_hz = np.asarray(frequencies, dtype=np.float64)

    if frequencies_hz.ndim != 1 or frequencies_hz.size == 0:
        raise ValueError(f"No frequencies found in {path}.")

    if not np.all(np.isfinite(frequencies_hz)):
        raise ValueError(f"Non-finite frequencies found in {path}.")

    return frequencies_hz


def find_frequency_bracket(
    frequencies_hz: NDArray[np.float64],
    target_frequency_hz: float,
) -> tuple[int, int, float]:
    """
    Find the tabulated frequencies bracketing a target frequency.

    Returns
    -------
    lower_index
        Index of the lower-frequency plane.

    upper_index
        Index of the upper-frequency plane.

    weight
        Linear interpolation weight applied to the upper-frequency plane.
        A value of zero selects the lower plane and a value of one selects
        the upper plane.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)

    if frequencies_hz.ndim != 1 or frequencies_hz.size == 0:
        raise ValueError("Frequency array must be non-empty and one-dimensional.")

    if not np.all(np.diff(frequencies_hz) > 0.0):
        raise ValueError("Frequencies must be strictly increasing.")

    if not (frequencies_hz[0] <= target_frequency_hz <= frequencies_hz[-1]):
        raise ValueError(
            f"Target frequency {target_frequency_hz / 1e6:.6f} MHz lies "
            f"outside the available range "
            f"{frequencies_hz[0] / 1e6:.6f}--"
            f"{frequencies_hz[-1] / 1e6:.6f} MHz."
        )

    exact_indices = np.flatnonzero(
        np.isclose(
            frequencies_hz,
            target_frequency_hz,
            rtol=0.0,
            atol=1e-6,
        )
    )

    if exact_indices.size:
        index = int(exact_indices[0])
        return index, index, 0.0

    upper_index = int(
        np.searchsorted(
            frequencies_hz,
            target_frequency_hz,
            side="right",
        )
    )
    lower_index = upper_index - 1

    lower_frequency_hz = frequencies_hz[lower_index]
    upper_frequency_hz = frequencies_hz[upper_index]

    weight = (target_frequency_hz - lower_frequency_hz) / (
        upper_frequency_hz - lower_frequency_hz
    )

    return lower_index, upper_index, float(weight)


def interpolate_complex(
    lower: NDArray[np.complex128],
    upper: NDArray[np.complex128],
    weight: float,
) -> NDArray[np.complex128]:
    """Linearly interpolate complex data through real and imaginary parts."""
    if lower.shape != upper.shape:
        raise ValueError(
            "Cannot interpolate arrays with different shapes: "
            f"{lower.shape} and {upper.shape}."
        )

    lower = np.asarray(lower, dtype=np.complex128)
    upper = np.asarray(upper, dtype=np.complex128)

    real = (1.0 - weight) * lower.real + weight * upper.real
    imag = (1.0 - weight) * lower.imag + weight * upper.imag

    return np.asarray(real + 1j * imag, dtype=np.complex128)


def remove_duplicate_azimuth_samples(
    az_rad: NDArray[np.float64],
    element_jones: NDArray[np.complex128],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
]:
    """
    Remove repeated azimuth endpoints after validating their Jones values.

    The source grid may contain both zero- and 360-degree samples. These
    correspond to the same physical direction and become duplicate azimuth
    samples after applying the modulo operation.
    """
    if az_rad.ndim != 1:
        raise ValueError("Azimuth coordinates must be one-dimensional.")

    if element_jones.shape[-1] != az_rad.size:
        raise ValueError(
            "The final Jones axis does not match the azimuth coordinate size."
        )

    duplicate_indices = np.flatnonzero(
        np.isclose(
            np.diff(az_rad),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
    )

    if duplicate_indices.size == 0:
        return az_rad, element_jones

    keep = np.ones(az_rad.size, dtype=bool)

    for lower_index in duplicate_indices:
        upper_index = lower_index + 1

        if not np.allclose(
            element_jones[..., lower_index],
            element_jones[..., upper_index],
            rtol=1e-5,
            atol=1e-8,
            equal_nan=True,
        ):
            difference = np.nanmax(
                np.abs(
                    element_jones[..., lower_index] - element_jones[..., upper_index]
                )
            )

            raise ValueError(
                "Duplicate azimuth samples contain inconsistent Jones values "
                f"at az={np.rad2deg(az_rad[lower_index]):.6f} degrees. "
                f"Maximum absolute difference: {difference:.6e}"
            )

        keep[upper_index] = False

    return az_rad[keep], element_jones[..., keep]


def load_element_jones_at_index(
    path: Path,
    frequency_index: int,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Load one average embedded-element Jones frequency plane.

    The returned azimuth convention is North equals zero, increasing eastward.

    Returns
    -------
    element_jones
        Complex array with shape ``(2, 2, n_za, n_az)``.

        Axis zero contains spherical vector components in the order
        ``("phi", "theta")``.

        Axis one contains feeds in the order ``("x", "y")``.

    za_rad
        Zenith angles in radians.

    az_rad
        Azimuths in radians, measured eastward from North.
    """
    with fits.open(path, memmap=True) as hdul:
        if not 0 <= frequency_index < len(hdul):
            raise IndexError(
                f"Frequency index {frequency_index} is outside {path}, "
                f"which has {len(hdul)} HDUs."
            )

        if hdul[frequency_index].data is None:
            raise ValueError(f"HDU {frequency_index} in {path} contains no data.")

        data = np.asarray(
            hdul[frequency_index].data,
            dtype=np.float64,
        )

    if data.ndim != 2 or data.shape[1] < 10:
        raise ValueError(
            "Expected a two-dimensional J-matrix table with at least "
            f"10 columns, got shape {data.shape}."
        )

    raw_theta_deg = np.asarray(data[:, 0], dtype=np.float64)
    raw_phi_deg = np.asarray(data[:, 1], dtype=np.float64)

    theta_deg = np.unique(raw_theta_deg)
    phi_deg = np.unique(raw_phi_deg)

    n_za = theta_deg.size
    n_phi = phi_deg.size

    if n_za * n_phi != raw_theta_deg.size:
        raise ValueError("J-matrix samples do not form a complete theta-phi grid.")

    expected_theta = np.tile(theta_deg, n_phi)
    expected_phi = np.repeat(phi_deg, n_za)

    if not np.allclose(
        raw_theta_deg,
        expected_theta,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("J-matrix theta samples are not in the expected ordering.")

    if not np.allclose(
        raw_phi_deg,
        expected_phi,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("J-matrix phi samples are not in the expected ordering.")

    element_jones = np.empty(
        (MWA_NFEED, MWA_NFEED, n_za, n_phi),
        dtype=np.complex128,
    )

    # Preserve the ordering used by the pyuvdata AEE implementation:
    #
    # element_jones[0, 0] = J_x_phi
    # element_jones[1, 0] = J_x_theta
    # element_jones[0, 1] = J_y_phi
    # element_jones[1, 1] = J_y_theta
    element_jones[1, 0] = (data[:, 2] + 1j * data[:, 3]).reshape(n_phi, n_za).T

    element_jones[0, 0] = (data[:, 4] + 1j * data[:, 5]).reshape(n_phi, n_za).T

    element_jones[1, 1] = (data[:, 6] + 1j * data[:, 7]).reshape(n_phi, n_za).T

    element_jones[0, 1] = (data[:, 8] + 1j * data[:, 9]).reshape(n_phi, n_za).T

    za_rad = np.deg2rad(theta_deg)

    # Source FEKO convention:
    #
    #   phi = 0 at East, increasing toward North.
    #
    # mwa_jaxbeam convention:
    #
    #   az = 0 at North, increasing eastward.
    #
    # Therefore:
    #
    #   az = (pi / 2 - phi) mod 2 pi
    phi_rad = np.deg2rad(phi_deg)
    az_rad = np.mod(np.pi / 2.0 - phi_rad, 2.0 * np.pi)

    az_order = np.argsort(az_rad)

    az_rad = az_rad[az_order]
    element_jones = element_jones[..., az_order]

    az_rad, element_jones = remove_duplicate_azimuth_samples(
        az_rad,
        element_jones,
    )

    if np.any(np.diff(az_rad) <= 0.0):
        raise ValueError("Converted azimuth coordinates are not strictly increasing.")

    return element_jones, za_rad, az_rad


def load_interpolated_element_jones(
    path: Path,
    lower_index: int,
    upper_index: int,
    weight: float,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Load and interpolate the J matrix to the target frequency."""
    lower_jones, lower_za_rad, lower_az_rad = load_element_jones_at_index(
        path, lower_index
    )

    if lower_index == upper_index:
        return lower_jones, lower_za_rad, lower_az_rad

    upper_jones, upper_za_rad, upper_az_rad = load_element_jones_at_index(
        path, upper_index
    )

    if not np.allclose(
        lower_za_rad,
        upper_za_rad,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Zenith-angle grids differ between the bracketing J-matrix HDUs."
        )

    if not np.allclose(
        lower_az_rad,
        upper_az_rad,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Azimuth grids differ between the bracketing J-matrix HDUs.")

    element_jones = interpolate_complex(
        lower_jones,
        upper_jones,
        weight,
    )

    return element_jones, lower_za_rad, lower_az_rad


def load_coupling_matrix_at_index(
    path: Path,
    frequency_index: int,
) -> NDArray[np.complex128]:
    """
    Load one complex 32-by-32 dipole coupling matrix.

    The FITS file orders its ports as Y0--Y15 followed by X0--X15. The
    returned matrix is reordered to X0--X15 followed by Y0--Y15.
    """
    with fits.open(path, memmap=True) as hdul:
        if not 0 <= frequency_index < len(hdul):
            raise IndexError(
                f"Frequency index {frequency_index} is outside {path}, "
                f"which has {len(hdul)} HDUs."
            )

        if hdul[frequency_index].data is None:
            raise ValueError(f"HDU {frequency_index} in {path} contains no data.")

        data = np.asarray(
            hdul[frequency_index].data,
            dtype=np.float64,
        )

    expected_shape = (2, MWA_NPORT, MWA_NPORT)

    if data.shape != expected_shape:
        raise ValueError(
            f"Expected Z-matrix data with shape {expected_shape}, got {data.shape}."
        )

    magnitude = data[0]
    phase_rad = data[1]

    coupling_yx = magnitude * np.exp(1j * phase_rad)

    n = MWA_NDIPOLE
    reorder = np.concatenate(
        (
            np.arange(n, 2 * n),
            np.arange(0, n),
        )
    )

    coupling_xy = coupling_yx[np.ix_(reorder, reorder)]

    return np.asarray(coupling_xy, dtype=np.complex128)


def load_interpolated_coupling_matrix(
    path: Path,
    lower_index: int,
    upper_index: int,
    weight: float,
) -> NDArray[np.complex128]:
    """Load and interpolate the complex Z matrix to the target frequency."""
    lower_matrix = load_coupling_matrix_at_index(
        path,
        lower_index,
    )

    if lower_index == upper_index:
        return lower_matrix

    upper_matrix = load_coupling_matrix_at_index(
        path,
        upper_index,
    )

    return interpolate_complex(
        lower_matrix,
        upper_matrix,
        weight,
    )


def interpolate_lna_impedance(
    path: Path,
    frequency_hz: float,
) -> complex:
    """Interpolate the measured complex LNA impedance."""
    data = np.genfromtxt(
        path,
        comments="#",
        dtype=np.float64,
    )

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"Expected at least three columns in {path}, got shape {data.shape}."
        )

    impedance_frequency_hz = np.asarray(
        data[:, 0],
        dtype=np.float64,
    )
    impedance_real_ohm = np.asarray(
        data[:, 1],
        dtype=np.float64,
    )
    impedance_imag_ohm = np.asarray(
        data[:, 2],
        dtype=np.float64,
    )

    finite = (
        np.isfinite(impedance_frequency_hz)
        & np.isfinite(impedance_real_ohm)
        & np.isfinite(impedance_imag_ohm)
    )

    impedance_frequency_hz = impedance_frequency_hz[finite]
    impedance_real_ohm = impedance_real_ohm[finite]
    impedance_imag_ohm = impedance_imag_ohm[finite]

    order = np.argsort(impedance_frequency_hz)

    impedance_frequency_hz = impedance_frequency_hz[order]
    impedance_real_ohm = impedance_real_ohm[order]
    impedance_imag_ohm = impedance_imag_ohm[order]

    if impedance_frequency_hz.size < 2:
        raise ValueError("At least two finite LNA impedance samples are required.")

    if not np.all(np.diff(impedance_frequency_hz) > 0.0):
        raise ValueError("LNA impedance frequencies must be strictly increasing.")

    if not (impedance_frequency_hz[0] <= frequency_hz <= impedance_frequency_hz[-1]):
        raise ValueError(
            f"Target frequency {frequency_hz / 1e6:.6f} MHz lies outside "
            f"the LNA impedance range "
            f"{impedance_frequency_hz[0] / 1e6:.6f}--"
            f"{impedance_frequency_hz[-1] / 1e6:.6f} MHz."
        )

    real_ohm = np.interp(
        frequency_hz,
        impedance_frequency_hz,
        impedance_real_ohm,
    )
    imag_ohm = np.interp(
        frequency_hz,
        impedance_frequency_hz,
        impedance_imag_ohm,
    )

    return complex(real_ohm, imag_ohm)


def make_dipole_positions_enu_m() -> NDArray[np.float64]:
    """
    Construct the nominal 4-by-4 MWA dipole positions.

    Returns
    -------
    positions_enu_m
        Array with shape ``(16, 3)`` in East-North-Up coordinates.
    """
    east_m, north_m = np.meshgrid(
        np.arange(4, dtype=np.float64) * MWA_DIPOLE_SPACING_M,
        np.flip(np.arange(4, dtype=np.float64) * MWA_DIPOLE_SPACING_M),
    )

    east_m = (east_m - east_m.mean()).reshape(-1)
    north_m = (north_m - north_m.mean()).reshape(-1)
    up_m = np.zeros(MWA_NDIPOLE, dtype=np.float64)

    return np.stack(
        (east_m, north_m, up_m),
        axis=-1,
    )


def validate_frequency_grids(
    j_frequencies_hz: NDArray[np.float64],
    z_frequencies_hz: NDArray[np.float64],
) -> None:
    """Validate that the J- and Z-matrix frequency grids agree."""
    if j_frequencies_hz.shape != z_frequencies_hz.shape:
        raise ValueError(
            "J- and Z-matrix files contain different numbers of "
            f"frequencies: {j_frequencies_hz.size} and "
            f"{z_frequencies_hz.size}."
        )

    if not np.allclose(
        j_frequencies_hz,
        z_frequencies_hz,
        rtol=0.0,
        atol=1e-6,
    ):
        maximum_difference_hz = np.max(np.abs(j_frequencies_hz - z_frequencies_hz))

        raise ValueError(
            "J- and Z-matrix frequency grids do not match. "
            f"Maximum difference: {maximum_difference_hz:.6f} Hz."
        )


def extract_aee_data(
    *,
    jmatrix_path: Path,
    zmatrix_path: Path,
    lna_impedance_path: Path,
    output_path: Path,
    target_frequency_hz: float,
) -> None:
    """Extract, interpolate and save the fixed-frequency AEE model."""
    download_if_missing(jmatrix_path, JMATRIX_URL)
    download_if_missing(zmatrix_path, ZMATRIX_URL)
    download_if_missing(lna_impedance_path, LNA_IMPEDANCE_URL)

    j_frequencies_hz = read_frequencies_hz(jmatrix_path)
    z_frequencies_hz = read_frequencies_hz(zmatrix_path)

    validate_frequency_grids(
        j_frequencies_hz,
        z_frequencies_hz,
    )

    lower_index, upper_index, interpolation_weight = find_frequency_bracket(
        j_frequencies_hz,
        target_frequency_hz,
    )

    lower_frequency_hz = float(j_frequencies_hz[lower_index])
    upper_frequency_hz = float(j_frequencies_hz[upper_index])

    print()
    print(f"Target frequency:      {target_frequency_hz / 1e6:.6f} MHz")

    if lower_index == upper_index:
        print(f"Exact FITS frequency:  {lower_frequency_hz / 1e6:.6f} MHz")
        print(f"FITS HDU index:        {lower_index}")
    else:
        print(f"Lower frequency:       {lower_frequency_hz / 1e6:.6f} MHz")
        print(f"Upper frequency:       {upper_frequency_hz / 1e6:.6f} MHz")
        print(f"Lower FITS HDU index:  {lower_index}")
        print(f"Upper FITS HDU index:  {upper_index}")
        print(f"Upper-plane weight:    {interpolation_weight:.10f}")

    element_jones, za_rad, az_rad = load_interpolated_element_jones(
        jmatrix_path,
        lower_index,
        upper_index,
        interpolation_weight,
    )

    coupling_matrix_ohm = load_interpolated_coupling_matrix(
        zmatrix_path,
        lower_index,
        upper_index,
        interpolation_weight,
    )

    lna_impedance_ohm = interpolate_lna_impedance(
        lna_impedance_path,
        target_frequency_hz,
    )

    z_total_ohm = (
        coupling_matrix_ohm
        + np.eye(
            MWA_NPORT,
            dtype=np.complex128,
        )
        * lna_impedance_ohm
    )

    dipole_positions_enu_m = make_dipole_positions_enu_m()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        frequency_hz=np.asarray(
            target_frequency_hz,
            dtype=np.float64,
        ),
        lower_frequency_hz=np.asarray(
            lower_frequency_hz,
            dtype=np.float64,
        ),
        upper_frequency_hz=np.asarray(
            upper_frequency_hz,
            dtype=np.float64,
        ),
        lower_frequency_index=np.asarray(
            lower_index,
            dtype=np.int64,
        ),
        upper_frequency_index=np.asarray(
            upper_index,
            dtype=np.int64,
        ),
        frequency_interpolation_weight=np.asarray(
            interpolation_weight,
            dtype=np.float64,
        ),
        az_rad=np.asarray(
            az_rad,
            dtype=np.float64,
        ),
        za_rad=np.asarray(
            za_rad,
            dtype=np.float64,
        ),
        element_jones=np.asarray(
            element_jones,
            dtype=np.complex128,
        ),
        coupling_matrix_ohm=np.asarray(
            coupling_matrix_ohm,
            dtype=np.complex128,
        ),
        lna_impedance_ohm=np.asarray(
            lna_impedance_ohm,
            dtype=np.complex128,
        ),
        z_total_ohm=np.asarray(
            z_total_ohm,
            dtype=np.complex128,
        ),
        dipole_positions_enu_m=np.asarray(
            dipole_positions_enu_m,
            dtype=np.float64,
        ),
        feed_order=np.asarray(
            ["x", "y"],
        ),
        vector_component_order=np.asarray(
            ["phi", "theta"],
        ),
        port_order=np.asarray(
            [
                *(f"x{index}" for index in range(MWA_NDIPOLE)),
                *(f"y{index}" for index in range(MWA_NDIPOLE)),
            ],
        ),
        coordinate_system=np.asarray(
            "ENU",
        ),
        azimuth_convention=np.asarray(
            "radians east of north; 0=north, pi/2=east",
        ),
        interpolation_method=np.asarray(
            "linear interpolation of real and imaginary components",
        ),
    )

    print()
    print(f"Saved archive:          {output_path}")
    print(f"Jones shape:            {element_jones.shape}")
    print(f"Coupling matrix shape:  {coupling_matrix_ohm.shape}")
    print(f"Z-total shape:          {z_total_ohm.shape}")
    print(f"Zenith-angle samples:   {za_rad.size}")
    print(f"Azimuth samples:        {az_rad.size}")
    print(
        "Azimuth range:          "
        f"{np.rad2deg(az_rad[0]):.6f}--"
        f"{np.rad2deg(az_rad[-1]):.6f} degrees"
    )
    print(
        "LNA impedance:          "
        f"{lna_impedance_ohm.real:.6f} "
        f"{lna_impedance_ohm.imag:+.6f}j ohm"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Extract and interpolate the MWA AEE model to 137 MHz.")
    )

    parser.add_argument(
        "--jmatrix",
        type=Path,
        default=DEFAULT_DATA_DIR / "Jmatrix.fits",
        help="Path to Jmatrix.fits.",
    )
    parser.add_argument(
        "--zmatrix",
        type=Path,
        default=DEFAULT_DATA_DIR / "ZMatrix.fits",
        help="Path to ZMatrix.fits.",
    )
    parser.add_argument(
        "--lna-impedance",
        type=Path,
        default=DEFAULT_DATA_DIR / "mwa_lna_impedance.txt",
        help="Path to mwa_lna_impedance.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output .npz path.",
    )
    parser.add_argument(
        "--target-frequency-hz",
        type=float,
        default=TARGET_FREQUENCY_HZ,
        help="Target frequency in Hz.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the extraction process."""
    args = parse_args()

    extract_aee_data(
        jmatrix_path=args.jmatrix,
        zmatrix_path=args.zmatrix,
        lna_impedance_path=args.lna_impedance,
        output_path=args.output,
        target_frequency_hz=args.target_frequency_hz,
    )


if __name__ == "__main__":
    main()
