"""
Extract and preprocess the MWA Average Embedded Element model at 137 MHz.

The script:

1. Ensures that the upstream reference files are available.
2. Finds the J- and Z-matrix frequencies bracketing 137 MHz.
3. Interpolates the complex J and Z matrices to exactly 137 MHz.
4. Interpolates the LNA impedance to exactly 137 MHz.
5. Converts the source azimuth convention to:

       az = 0 at North, increasing eastward.

6. Saves the arrays required by the JAX runtime model.

The interpolation and preprocessing are performed in double precision. Arrays
written to the runtime archive use float32 and complex64.

Run from the repository root with:

    uv run python data/extract_aee_137mhz.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from mwa_jaxbeam.reference_data import (
    DEFAULT_REFERENCE_DIR,
    ensure_reference_data,
)


TARGET_FREQUENCY_HZ = 137e6

MWA_DIPOLE_SPACING_M = 1.1
MWA_NFEED = 2
MWA_NDIPOLE = 16
MWA_NPORT = MWA_NFEED * MWA_NDIPOLE

DEFAULT_OUTPUT_PATH = (
    DEFAULT_REFERENCE_DIR.parent / "src" / "mwa_jaxbeam" / "data" / "aee_137mhz.npz"
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def read_frequencies_hz(path: Path) -> FloatArray:
    """Read the frequency associated with each FITS HDU."""
    frequencies: list[float] = []

    with fits.open(path, memmap=True) as hdul:
        for hdu_index, hdu in enumerate(hdul):
            if "FREQ" not in hdu.header:
                raise KeyError(f"HDU {hdu_index} in {path} has no FREQ header.")

            frequencies.append(float(hdu.header["FREQ"]))

    frequencies_hz = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    if frequencies_hz.ndim != 1 or frequencies_hz.size == 0:
        raise ValueError(f"No frequencies found in {path}.")

    if not np.all(np.isfinite(frequencies_hz)):
        raise ValueError(f"Non-finite frequencies found in {path}.")

    return frequencies_hz


def validate_frequency_grids(
    j_frequencies_hz: FloatArray,
    z_frequencies_hz: FloatArray,
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
        maximum_difference_hz = float(
            np.max(np.abs(j_frequencies_hz - z_frequencies_hz))
        )

        raise ValueError(
            "J- and Z-matrix frequency grids do not match. "
            f"Maximum difference: {maximum_difference_hz:.6f} Hz."
        )


def find_frequency_bracket(
    frequencies_hz: FloatArray,
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
    """
    frequencies_hz = np.asarray(
        frequencies_hz,
        dtype=np.float64,
    )

    if frequencies_hz.ndim != 1 or frequencies_hz.size == 0:
        raise ValueError("Frequency array must be non-empty and one-dimensional.")

    if not np.all(np.isfinite(frequencies_hz)):
        raise ValueError("Frequency array contains non-finite values.")

    if not np.all(np.diff(frequencies_hz) > 0.0):
        raise ValueError("Frequencies must be strictly increasing.")

    target_frequency_hz = float(target_frequency_hz)

    if not (frequencies_hz[0] <= target_frequency_hz <= frequencies_hz[-1]):
        raise ValueError(
            f"Target frequency "
            f"{target_frequency_hz / 1e6:.6f} MHz lies "
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
    lower: ComplexArray,
    upper: ComplexArray,
    weight: float,
) -> ComplexArray:
    """Linearly interpolate complex values component-wise."""
    lower = np.asarray(lower, dtype=np.complex128)
    upper = np.asarray(upper, dtype=np.complex128)

    if lower.shape != upper.shape:
        raise ValueError(
            "Cannot interpolate arrays with different shapes: "
            f"{lower.shape} and {upper.shape}."
        )

    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"Interpolation weight must lie in [0, 1], got {weight}.")

    return (
        (1.0 - weight) * lower.real
        + weight * upper.real
        + 1j * ((1.0 - weight) * lower.imag + weight * upper.imag)
    )


def validate_jmatrix_table(
    data: FloatArray,
    *,
    path: Path,
) -> tuple[FloatArray, FloatArray]:
    """
    Validate the J-matrix sample grid.

    Returns
    -------
    theta_deg
        Unique zenith-angle samples in degrees.
    phi_deg
        Unique FEKO azimuth samples in degrees.
    """
    if data.ndim != 2 or data.shape[1] < 10:
        raise ValueError(
            "Expected a two-dimensional J-matrix table with at least "
            f"10 columns, got shape {data.shape} from {path}."
        )

    raw_theta_deg = np.asarray(
        data[:, 0],
        dtype=np.float64,
    )
    raw_phi_deg = np.asarray(
        data[:, 1],
        dtype=np.float64,
    )

    if not (np.all(np.isfinite(raw_theta_deg)) and np.all(np.isfinite(raw_phi_deg))):
        raise ValueError(f"Non-finite angular coordinates found in {path}.")

    theta_deg = np.unique(raw_theta_deg)
    phi_deg = np.unique(raw_phi_deg)

    n_za = theta_deg.size
    n_phi = phi_deg.size

    if n_za * n_phi != raw_theta_deg.size:
        raise ValueError("J-matrix samples do not form a complete theta-phi grid.")

    expected_theta_deg = np.tile(
        theta_deg,
        n_phi,
    )
    expected_phi_deg = np.repeat(
        phi_deg,
        n_za,
    )

    if not np.allclose(
        raw_theta_deg,
        expected_theta_deg,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("J-matrix theta samples are not in the expected ordering.")

    if not np.allclose(
        raw_phi_deg,
        expected_phi_deg,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("J-matrix phi samples are not in the expected ordering.")

    return theta_deg, phi_deg


def construct_element_jones(
    data: FloatArray,
    *,
    n_za: int,
    n_phi: int,
) -> ComplexArray:
    """
    Construct the embedded-element Jones array from a FITS table.

    Returns
    -------
    element_jones
        Complex array with shape ``(2, 2, n_za, n_phi)``.

        Axis zero contains spherical vector components in the order
        ``("phi", "theta")``.

        Axis one contains feeds in the order ``("x", "y")``.
    """
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

    return element_jones


def convert_feko_azimuth(
    phi_deg: FloatArray,
    element_jones: ComplexArray,
) -> tuple[FloatArray, ComplexArray]:
    """
    Convert FEKO azimuths to radians east of North.

    The source FEKO convention is

        phi = 0 at East, increasing toward North.

    The ``mwa_jaxbeam`` convention is

        az = 0 at North, increasing eastward.

    Therefore

        az = (pi / 2 - phi) mod 2 pi.
    """
    phi_rad = np.deg2rad(
        np.asarray(
            phi_deg,
            dtype=np.float64,
        )
    )

    az_rad = np.mod(
        np.pi / 2.0 - phi_rad,
        2.0 * np.pi,
    )

    az_order = np.argsort(az_rad)

    return (
        az_rad[az_order],
        element_jones[..., az_order],
    )


def remove_duplicate_azimuth_samples(
    az_rad: FloatArray,
    element_jones: ComplexArray,
) -> tuple[FloatArray, ComplexArray]:
    """
    Remove repeated azimuth endpoints after validating their Jones values.

    The source grid may contain both zero- and 360-degree samples. They
    represent the same physical direction and become duplicate samples after
    the azimuth conversion and sorting.
    """
    az_rad = np.asarray(
        az_rad,
        dtype=np.float64,
    )
    element_jones = np.asarray(
        element_jones,
        dtype=np.complex128,
    )

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

    keep = np.ones(
        az_rad.size,
        dtype=np.bool_,
    )

    for lower_index in duplicate_indices:
        upper_index = int(lower_index + 1)

        lower_jones = element_jones[..., lower_index]
        upper_jones = element_jones[..., upper_index]

        if not np.allclose(
            lower_jones,
            upper_jones,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=True,
        ):
            difference = float(np.nanmax(np.abs(lower_jones - upper_jones)))

            raise ValueError(
                "Duplicate azimuth samples contain inconsistent "
                "Jones values at "
                f"az={np.rad2deg(az_rad[lower_index]):.6f} degrees. "
                f"Maximum absolute difference: {difference:.6e}"
            )

        keep[upper_index] = False

    return (
        az_rad[keep],
        element_jones[..., keep],
    )


def load_element_jones_at_index(
    path: Path,
    frequency_index: int,
) -> tuple[ComplexArray, FloatArray, FloatArray]:
    """
    Load one average embedded-element Jones frequency plane.

    Returns
    -------
    element_jones
        Complex array with shape ``(2, 2, n_za, n_az)``.
    za_rad
        Zenith angles in radians.
    az_rad
        Azimuths in radians east of North.
    """
    with fits.open(path, memmap=True) as hdul:
        if not 0 <= frequency_index < len(hdul):
            raise IndexError(
                f"Frequency index {frequency_index} is outside "
                f"{path}, which has {len(hdul)} HDUs."
            )

        hdu = hdul[frequency_index]

        if hdu.data is None:
            raise ValueError(f"HDU {frequency_index} in {path} contains no data.")

        data = np.asarray(
            hdu.data,
            dtype=np.float64,
        )

    theta_deg, phi_deg = validate_jmatrix_table(
        data,
        path=path,
    )

    element_jones = construct_element_jones(
        data,
        n_za=theta_deg.size,
        n_phi=phi_deg.size,
    )

    az_rad, element_jones = convert_feko_azimuth(
        phi_deg,
        element_jones,
    )

    az_rad, element_jones = remove_duplicate_azimuth_samples(
        az_rad,
        element_jones,
    )

    if np.any(np.diff(az_rad) <= 0.0):
        raise ValueError("Converted azimuth coordinates are not strictly increasing.")

    za_rad = np.deg2rad(theta_deg)

    return element_jones, za_rad, az_rad


def load_interpolated_element_jones(
    path: Path,
    lower_index: int,
    upper_index: int,
    weight: float,
) -> tuple[ComplexArray, FloatArray, FloatArray]:
    """Load and interpolate the J matrix to 137 MHz."""
    lower_jones, lower_za_rad, lower_az_rad = load_element_jones_at_index(
        path,
        lower_index,
    )

    if lower_index == upper_index:
        return lower_jones, lower_za_rad, lower_az_rad

    upper_jones, upper_za_rad, upper_az_rad = load_element_jones_at_index(
        path,
        upper_index,
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
) -> ComplexArray:
    """
    Load one complex 32-by-32 dipole coupling matrix.

    The FITS file orders ports as Y0--Y15 followed by X0--X15. The returned
    matrix is reordered to X0--X15 followed by Y0--Y15.
    """
    with fits.open(path, memmap=True) as hdul:
        if not 0 <= frequency_index < len(hdul):
            raise IndexError(
                f"Frequency index {frequency_index} is outside "
                f"{path}, which has {len(hdul)} HDUs."
            )

        hdu = hdul[frequency_index]

        if hdu.data is None:
            raise ValueError(f"HDU {frequency_index} in {path} contains no data.")

        data = np.asarray(
            hdu.data,
            dtype=np.float64,
        )

    expected_shape = (
        2,
        MWA_NPORT,
        MWA_NPORT,
    )

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

    return coupling_yx[np.ix_(reorder, reorder)]


def load_interpolated_coupling_matrix(
    path: Path,
    lower_index: int,
    upper_index: int,
    weight: float,
) -> ComplexArray:
    """Load and interpolate the Z matrix to 137 MHz."""
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

    impedance_frequency_hz = data[:, 0]
    impedance_real_ohm = data[:, 1]
    impedance_imag_ohm = data[:, 2]

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

    frequency_hz = float(frequency_hz)

    if not (impedance_frequency_hz[0] <= frequency_hz <= impedance_frequency_hz[-1]):
        raise ValueError(
            f"Target frequency {frequency_hz / 1e6:.6f} MHz "
            f"lies outside the LNA impedance range "
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


def make_dipole_positions_enu_m() -> FloatArray:
    """
    Construct the nominal 4-by-4 MWA dipole positions.

    Returns
    -------
    positions_enu_m
        Array with shape ``(16, 3)`` in East-North-Up coordinates.
    """
    offsets_m = np.arange(4, dtype=np.float64) * MWA_DIPOLE_SPACING_M

    east_m, north_m = np.meshgrid(
        offsets_m,
        np.flip(offsets_m),
    )

    east_m = (east_m - east_m.mean()).reshape(-1)
    north_m = (north_m - north_m.mean()).reshape(-1)
    up_m = np.zeros(MWA_NDIPOLE, dtype=np.float64)

    return np.stack(
        (
            east_m,
            north_m,
            up_m,
        ),
        axis=-1,
    )


def save_runtime_archive(
    *,
    output_path: Path,
    az_rad: FloatArray,
    za_rad: FloatArray,
    element_jones: ComplexArray,
    z_total_ohm: ComplexArray,
    dipole_positions_enu_m: FloatArray,
) -> None:
    """Save only the arrays required by the 137 MHz JAX runtime."""
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_path,
        az_rad=np.asarray(
            az_rad,
            dtype=np.float32,
        ),
        za_rad=np.asarray(
            za_rad,
            dtype=np.float32,
        ),
        element_jones=np.asarray(
            element_jones,
            dtype=np.complex64,
        ),
        z_total_ohm=np.asarray(
            z_total_ohm,
            dtype=np.complex64,
        ),
        dipole_positions_enu_m=np.asarray(
            dipole_positions_enu_m,
            dtype=np.float32,
        ),
    )


def extract_aee_data(
    *,
    jmatrix_path: Path,
    zmatrix_path: Path,
    lna_impedance_path: Path,
    output_path: Path,
) -> None:
    """Extract, interpolate, and save the 137 MHz AEE model."""
    j_frequencies_hz = read_frequencies_hz(jmatrix_path)
    z_frequencies_hz = read_frequencies_hz(zmatrix_path)

    validate_frequency_grids(
        j_frequencies_hz,
        z_frequencies_hz,
    )

    (
        lower_index,
        upper_index,
        interpolation_weight,
    ) = find_frequency_bracket(
        j_frequencies_hz,
        TARGET_FREQUENCY_HZ,
    )

    lower_frequency_hz = float(j_frequencies_hz[lower_index])
    upper_frequency_hz = float(j_frequencies_hz[upper_index])

    print()
    print(f"Target frequency:      {TARGET_FREQUENCY_HZ / 1e6:.6f} MHz")

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
        TARGET_FREQUENCY_HZ,
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

    save_runtime_archive(
        output_path=output_path,
        az_rad=az_rad,
        za_rad=za_rad,
        element_jones=element_jones,
        z_total_ohm=z_total_ohm,
        dipole_positions_enu_m=dipole_positions_enu_m,
    )

    print()
    print(f"Saved archive:          {output_path}")
    print(f"Jones shape:            {element_jones.shape}")
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
    print("Runtime real dtype:     float32")
    print("Runtime complex dtype:  complex64")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Extract and interpolate the MWA AEE model to 137 MHz.")
    )

    parser.add_argument(
        "--jmatrix",
        type=Path,
        default=DEFAULT_REFERENCE_DIR / "Jmatrix.fits",
        help=(
            "Path to Jmatrix.fits. Missing files are downloaded. "
            f"Default: {DEFAULT_REFERENCE_DIR / 'Jmatrix.fits'}"
        ),
    )
    parser.add_argument(
        "--zmatrix",
        type=Path,
        default=DEFAULT_REFERENCE_DIR / "ZMatrix.fits",
        help=(
            "Path to ZMatrix.fits. Missing files are downloaded. "
            f"Default: {DEFAULT_REFERENCE_DIR / 'ZMatrix.fits'}"
        ),
    )
    parser.add_argument(
        "--lna-impedance",
        type=Path,
        default=(DEFAULT_REFERENCE_DIR / "mwa_lna_impedance.txt"),
        help=(
            "Path to mwa_lna_impedance.txt. Missing files are "
            "downloaded. "
            f"Default: "
            f"{DEFAULT_REFERENCE_DIR / 'mwa_lna_impedance.txt'}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(f"Output .npz path. Default: {DEFAULT_OUTPUT_PATH}"),
    )

    return parser.parse_args()


def main() -> None:
    """Run the extraction process."""
    args = parse_args()

    ensure_reference_data(
        jmatrix_path=args.jmatrix,
        zmatrix_path=args.zmatrix,
        lna_impedance_path=args.lna_impedance,
    )

    extract_aee_data(
        jmatrix_path=args.jmatrix,
        zmatrix_path=args.zmatrix,
        lna_impedance_path=args.lna_impedance,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
