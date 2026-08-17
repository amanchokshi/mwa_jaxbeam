"""
Differentiable MWA Average Embedded Element beam model at 137 MHz.

Azimuth is measured eastward from North:

    az = 0       North
    az = pi / 2  East
    az = pi      South
    az = 3pi / 2 West

Zenith angle is measured downward from zenith:

    za = 0       Zenith
    za = pi / 2  Horizon

The model separates two per-dipole controls:

- ``dipole_flags`` are boolean participation flags applied to the port-driving
  voltages before the mutually coupled impedance solve;
- ``dipole_gains`` are real or complex post-current gains applied to each
  driven-feed/dipole contribution before the geometric array-factor sum.

With all dipoles enabled and unity gains, the model reduces to the nominal AEE
beam.

The returned Jones matrix has shape::

    (2, 2, *broadcast_shape)

where:

- axis 0 contains the sky-vector components ``(phi, theta)``;
- axis 1 contains the driven tile feeds ``(X, Y)``.

also note that the dipole orientation convention was inherited
from pyuvdata which we compare to:

- XX: East–West dipoles (X feed)
- YY: North–South dipoles (Y feed)
"""

from __future__ import annotations

from importlib.resources import as_file, files
from typing import Final

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

__all__ = [
    "AZIMUTH_RAD",
    "DIPOLE_POSITIONS_ENU_M",
    "ELEMENT_JONES",
    "FREQUENCY_HZ",
    "N_DIPOLES",
    "N_FEEDS",
    "N_PORTS",
    "WAVENUMBER_RAD_PER_M",
    "ZA_RAD",
    "Z_TOTAL_OHM",
    "array_factor",
    "coherency",
    "direction_enu",
    "element_jones",
    "jones",
    "port_currents",
    "power",
]


FREQUENCY_HZ: Final[float] = 137e6

N_FEEDS: Final[int] = 2
N_DIPOLES: Final[int] = 16
N_PORTS: Final[int] = N_FEEDS * N_DIPOLES

SPEED_OF_LIGHT_M_PER_S: Final[float] = 299_792_458.0
TWO_PI: Final[float] = 2.0 * np.pi

REAL_DTYPE = jnp.float32
COMPLEX_DTYPE = jnp.complex64


def _load_model_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load the arrays required by the packaged 137 MHz AEE model."""
    resource = files("mwa_jaxbeam").joinpath(
        "data",
        "aee_137mhz.npz",
    )

    with as_file(resource) as path:
        with np.load(path) as archive:
            return (
                np.array(archive["az_rad"], copy=True),
                np.array(archive["za_rad"], copy=True),
                np.array(archive["element_jones"], copy=True),
                np.array(archive["z_total_ohm"], copy=True),
                np.array(
                    archive["dipole_positions_enu_m"],
                    copy=True,
                ),
            )


(
    _azimuth_rad,
    _za_rad,
    _element_jones,
    _z_total_ohm,
    _dipole_positions_enu_m,
) = _load_model_data()


AZIMUTH_RAD: Final[Array] = jnp.asarray(
    _azimuth_rad,
    dtype=REAL_DTYPE,
)

ZA_RAD: Final[Array] = jnp.asarray(
    _za_rad,
    dtype=REAL_DTYPE,
)

ELEMENT_JONES: Final[Array] = jnp.asarray(
    _element_jones,
    dtype=COMPLEX_DTYPE,
)

Z_TOTAL_OHM: Final[Array] = jnp.asarray(
    _z_total_ohm,
    dtype=COMPLEX_DTYPE,
)

DIPOLE_POSITIONS_ENU_M: Final[Array] = jnp.asarray(
    _dipole_positions_enu_m,
    dtype=REAL_DTYPE,
)

del (
    _azimuth_rad,
    _za_rad,
    _element_jones,
    _z_total_ohm,
    _dipole_positions_enu_m,
)


WAVENUMBER_RAD_PER_M: Final[Array] = jnp.asarray(
    TWO_PI * FREQUENCY_HZ / SPEED_OF_LIGHT_M_PER_S,
    dtype=REAL_DTYPE,
)

_DEFAULT_DIPOLE_FLAGS: Final[Array] = jnp.ones(
    (N_FEEDS, N_DIPOLES),
    dtype=jnp.bool_,
)

_DEFAULT_DIPOLE_GAINS: Final[Array] = jnp.ones(
    (N_FEEDS, N_DIPOLES),
    dtype=COMPLEX_DTYPE,
)


def _validate_model_data() -> None:
    """Validate the dimensions and coordinate grids of the model archive."""
    expected_jones_shape = (
        N_FEEDS,
        N_FEEDS,
        ZA_RAD.size,
        AZIMUTH_RAD.size,
    )

    if ELEMENT_JONES.shape != expected_jones_shape:
        raise ValueError(f"Unexpected element-Jones shape. Expected {expected_jones_shape}, got {ELEMENT_JONES.shape}.")

    expected_impedance_shape = (
        N_PORTS,
        N_PORTS,
    )

    if Z_TOTAL_OHM.shape != expected_impedance_shape:
        raise ValueError(
            f"Unexpected total-impedance shape. Expected {expected_impedance_shape}, got {Z_TOTAL_OHM.shape}."
        )

    expected_position_shape = (
        N_DIPOLES,
        3,
    )

    if DIPOLE_POSITIONS_ENU_M.shape != expected_position_shape:
        raise ValueError(
            f"Unexpected dipole-position shape. Expected {expected_position_shape}, got {DIPOLE_POSITIONS_ENU_M.shape}."
        )

    azimuth_rad = np.asarray(AZIMUTH_RAD)
    zenith_angle_rad = np.asarray(ZA_RAD)

    if not np.all(np.isfinite(azimuth_rad)):
        raise ValueError("The model azimuth coordinates contain non-finite values.")

    if not np.all(np.isfinite(zenith_angle_rad)):
        raise ValueError("The model zenith-angle coordinates contain non-finite values.")

    if not np.all(np.diff(azimuth_rad) > 0.0):
        raise ValueError("The model azimuth coordinates must be strictly increasing.")

    if not np.all(np.diff(zenith_angle_rad) > 0.0):
        raise ValueError("The model zenith-angle coordinates must be strictly increasing.")


_validate_model_data()


def _as_dipole_flags(
    dipole_flags: ArrayLike | None,
) -> Array:
    """
    Convert dipole flags to boolean shape ``(2, 16)``.

    Accepted inputs are:

    - ``None``: every dipole is enabled;
    - boolean scalar: broadcast to all dipoles and both feeds;
    - boolean shape ``(16,)``: copied across both feeds;
    - boolean shape ``(2, 16)``: separate X and Y flags.
    """
    if dipole_flags is None:
        return _DEFAULT_DIPOLE_FLAGS

    flag_array = jnp.asarray(dipole_flags)

    if not jnp.issubdtype(flag_array.dtype, jnp.bool_):
        raise ValueError(f"dipole_flags must contain boolean values; got dtype {flag_array.dtype}.")

    if flag_array.ndim == 0:
        return jnp.full(
            (N_FEEDS, N_DIPOLES),
            flag_array,
            dtype=jnp.bool_,
        )

    if flag_array.shape == (N_DIPOLES,):
        return jnp.broadcast_to(
            flag_array[jnp.newaxis, :],
            (N_FEEDS, N_DIPOLES),
        )

    if flag_array.shape == (N_FEEDS, N_DIPOLES):
        return flag_array

    raise ValueError(
        "dipole_flags must be a boolean scalar, have shape "
        f"{(N_DIPOLES,)}, or have shape "
        f"{(N_FEEDS, N_DIPOLES)}; got "
        f"{flag_array.shape}."
    )


def _as_dipole_gains(
    dipole_gains: ArrayLike | None,
) -> Array:
    """
    Convert post-current dipole gains to complex shape ``(2, 16)``.

    Accepted inputs are:

    - ``None``: unity gain for every dipole and both feeds;
    - scalar: broadcast to all dipoles and both feeds;
    - shape ``(16,)``: copied across both feeds;
    - shape ``(2, 16)``: separate X and Y gains.

    Real values represent amplitude-only gains. Complex values additionally
    represent phase offsets.
    """
    if dipole_gains is None:
        return _DEFAULT_DIPOLE_GAINS

    gain_array = jnp.asarray(
        dipole_gains,
        dtype=COMPLEX_DTYPE,
    )

    if gain_array.ndim == 0:
        return jnp.full(
            (N_FEEDS, N_DIPOLES),
            gain_array,
            dtype=COMPLEX_DTYPE,
        )

    if gain_array.shape == (N_DIPOLES,):
        return jnp.broadcast_to(
            gain_array[jnp.newaxis, :],
            (N_FEEDS, N_DIPOLES),
        )

    if gain_array.shape == (N_FEEDS, N_DIPOLES):
        return gain_array

    raise ValueError(
        "dipole_gains must be a scalar, have shape "
        f"{(N_DIPOLES,)}, or have shape "
        f"{(N_FEEDS, N_DIPOLES)}; got "
        f"{gain_array.shape}."
    )


def direction_enu(
    az_rad: ArrayLike,
    za_rad: ArrayLike,
) -> Array:
    """
    Convert azimuth and zenith angle to ENU direction cosines.

    Parameters
    ----------
    az_rad
        Azimuth in radians, measured eastward from North.
    za_rad
        Zenith angle in radians.

    Returns
    -------
    direction
        Direction cosines with shape ``(*broadcast_shape, 3)``.
        The final axis contains ``(East, North, Up)``.

    Notes
    -----
    For North-through-East azimuth, the ENU unit vector is

    ``(sin(za) sin(az), sin(za) cos(az), cos(za))``.
    """
    azimuth = jnp.asarray(
        az_rad,
        dtype=REAL_DTYPE,
    )
    zenith_angle = jnp.asarray(
        za_rad,
        dtype=REAL_DTYPE,
    )

    azimuth, zenith_angle = jnp.broadcast_arrays(
        azimuth,
        zenith_angle,
    )

    sin_za = jnp.sin(zenith_angle)

    east = sin_za * jnp.sin(azimuth)
    north = sin_za * jnp.cos(azimuth)
    up = jnp.cos(zenith_angle)

    return jnp.stack(
        (east, north, up),
        axis=-1,
    )


def _periodic_azimuth_indices(
    az_rad: Array,
) -> tuple[Array, Array, Array]:
    """
    Find the neighbouring periodic azimuth samples.

    Returns the lower index, upper index, and fractional distance from the
    lower sample to the upper sample.
    """
    azimuth = jnp.mod(
        az_rad,
        TWO_PI,
    )

    upper_index = jnp.searchsorted(
        AZIMUTH_RAD,
        azimuth,
        side="right",
    )

    lower_index = jnp.mod(
        upper_index - 1,
        AZIMUTH_RAD.size,
    )
    upper_index = jnp.mod(
        upper_index,
        AZIMUTH_RAD.size,
    )

    lower_azimuth = AZIMUTH_RAD[lower_index]
    upper_azimuth = AZIMUTH_RAD[upper_index]

    # The upper neighbour of the final grid point is the first point
    # interpreted as 2*pi rather than zero.
    upper_azimuth = jnp.where(
        upper_index == 0,
        upper_azimuth + TWO_PI,
        upper_azimuth,
    )

    # This handles values that lie on the wrapped side of the grid.
    evaluation_azimuth = jnp.where(
        azimuth < lower_azimuth,
        azimuth + TWO_PI,
        azimuth,
    )

    weight = (evaluation_azimuth - lower_azimuth) / (upper_azimuth - lower_azimuth)

    return lower_index, upper_index, weight


def _zenith_angle_indices(
    za_rad: Array,
) -> tuple[Array, Array, Array]:
    """
    Find the neighbouring zenith-angle samples.

    Zenith angles outside the tabulated range are clipped to the nearest
    boundary.
    """
    zenith_angle = jnp.clip(
        za_rad,
        ZA_RAD[0],
        ZA_RAD[-1],
    )

    upper_index = jnp.searchsorted(
        ZA_RAD,
        zenith_angle,
        side="right",
    )

    upper_index = jnp.clip(
        upper_index,
        1,
        ZA_RAD.size - 1,
    )
    lower_index = upper_index - 1

    lower_za = ZA_RAD[lower_index]
    upper_za = ZA_RAD[upper_index]

    weight = (zenith_angle - lower_za) / (upper_za - lower_za)

    return lower_index, upper_index, weight


def element_jones(
    az_rad: ArrayLike,
    za_rad: ArrayLike,
) -> Array:
    """
    Interpolate the average embedded-element Jones matrix.

    Bilinear interpolation is performed directly on the complex Jones
    values. This is equivalent to independently interpolating their real and
    imaginary components.

    Azimuth interpolation is periodic. Zenith angles outside the model grid
    are clipped to its nearest boundary.

    Parameters
    ----------
    az_rad
        Azimuth in radians, measured eastward from North.
    za_rad
        Zenith angle in radians.

    Returns
    -------
    jones_element
        Interpolated element Jones matrix with shape
        ``(2, 2, *broadcast_shape)``.

        Axis zero contains ``(phi, theta)`` and axis one contains ``(X, Y)``.
    """
    azimuth = jnp.asarray(
        az_rad,
        dtype=REAL_DTYPE,
    )
    zenith_angle = jnp.asarray(
        za_rad,
        dtype=REAL_DTYPE,
    )

    azimuth, zenith_angle = jnp.broadcast_arrays(
        azimuth,
        zenith_angle,
    )

    az0, az1, az_weight = _periodic_azimuth_indices(azimuth)
    za0, za1, za_weight = _zenith_angle_indices(zenith_angle)

    j_za0_az0 = ELEMENT_JONES[..., za0, az0]
    j_za0_az1 = ELEMENT_JONES[..., za0, az1]
    j_za1_az0 = ELEMENT_JONES[..., za1, az0]
    j_za1_az1 = ELEMENT_JONES[..., za1, az1]

    # Add the two leading Jones axes to the interpolation weights.
    az_weight = az_weight[jnp.newaxis, jnp.newaxis, ...]
    za_weight = za_weight[jnp.newaxis, jnp.newaxis, ...]

    # Interpolate in azimuth at the lower zenith-angle sample.
    jones_at_za0 = (1.0 - az_weight) * j_za0_az0 + az_weight * j_za0_az1

    # Interpolate in azimuth at the upper zenith-angle sample.
    jones_at_za1 = (1.0 - az_weight) * j_za1_az0 + az_weight * j_za1_az1

    # Interpolate between the two zenith-angle samples.
    return (1.0 - za_weight) * jones_at_za0 + za_weight * jones_at_za1


def port_currents(
    dipole_flags: ArrayLike | None = None,
) -> Array:
    """
    Calculate the mutually coupled currents for each driven feed.

    Parameters
    ----------
    dipole_flags
        Boolean flags controlling which dipole ports are driven.

        Accepted forms are:

        - ``None``: all dipoles enabled;
        - boolean scalar: applied to every dipole and both feeds;
        - shape ``(16,)``: copied across both feeds;
        - shape ``(2, 16)``: separate X and Y flags.

        Flags are applied to the port-driving voltages before solving the
        mutually coupled impedance system.

    Returns
    -------
    currents
        Complex currents with shape ``(2, 2, 16)``.

        The axes are:

        1. port polarization ``(X, Y)``;
        2. driven tile feed ``(X, Y)``;
        3. dipole number.

        Off-diagonal values describe currents induced in one polarization
        when the other feed is driven.

    Notes
    -----
    The current response satisfies

    ``Z_total @ current = excitation_voltage``.

    Each feed is driven independently. For the X-feed solution, enabled X
    ports are driven with unit voltage and all Y-port voltages are zero. For
    the Y-feed solution, enabled Y ports are driven with unit voltage and all
    X-port voltages are zero.
    """
    flag_array = _as_dipole_flags(dipole_flags)

    # Each column is an independent driven-feed problem:
    #
    # column 0: drive the enabled 16 X ports with unit voltage
    # column 1: drive the enabled 16 Y ports with unit voltage
    excitation_voltage = jnp.zeros(
        (N_PORTS, N_FEEDS),
        dtype=COMPLEX_DTYPE,
    )

    excitation_voltage = excitation_voltage.at[
        :N_DIPOLES,
        0,
    ].set(flag_array[0].astype(COMPLEX_DTYPE))

    excitation_voltage = excitation_voltage.at[
        N_DIPOLES:,
        1,
    ].set(flag_array[1].astype(COMPLEX_DTYPE))

    currents = jnp.linalg.solve(
        Z_TOTAL_OHM,
        excitation_voltage,
    )

    currents = currents.reshape(
        N_FEEDS,
        N_DIPOLES,
        N_FEEDS,
    )

    return jnp.swapaxes(
        currents,
        1,
        2,
    )


def array_factor(
    az_rad: ArrayLike,
    za_rad: ArrayLike,
    dipole_flags: ArrayLike | None = None,
    dipole_gains: ArrayLike | None = None,
) -> Array:
    """
    Calculate the coupled array-factor matrix.

    Parameters
    ----------
    az_rad
        Azimuth in radians, measured eastward from North.
    za_rad
        Zenith angle in radians.
    dipole_flags
        Boolean dipole flags applied before the mutually coupled current
        solve. See :func:`port_currents`.
    dipole_gains
        Real or complex post-current gains. Gains are indexed by driven feed
        and dipole, with shape ``(2, 16)`` after broadcasting. For a given
        driven feed and dipole, the same gain multiplies both port-polarization
        current components produced by the coupled solve.

    Returns
    -------
    factor
        Complex array-factor matrix with shape
        ``(2, 2, *broadcast_shape)``.

        Axis zero is the port polarization ``(X, Y)`` and axis one is the
        independently driven feed ``(X, Y)``.
    """
    direction = direction_enu(
        az_rad,
        za_rad,
    )

    geometric_path_m = jnp.sum(
        direction[..., jnp.newaxis, :] * DIPOLE_POSITIONS_ENU_M,
        axis=-1,
    )

    geometric_phase = jnp.exp(1j * WAVENUMBER_RAD_PER_M * geometric_path_m)

    currents = port_currents(dipole_flags=dipole_flags)
    gain_array = _as_dipole_gains(dipole_gains)

    # currents has shape:
    #     (port polarization, driven feed, dipole)
    # gain_array has shape:
    #     (driven feed, dipole)
    #
    # A gain therefore scales the pair of X/Y port currents associated with
    # one driven-feed/dipole entry, while leaving the mutual-coupling solve
    # itself unchanged.
    effective_currents = currents * gain_array[jnp.newaxis, :, :]

    weighted_currents = (
        geometric_phase[
            ...,
            jnp.newaxis,
            jnp.newaxis,
            :,
        ]
        * effective_currents
    )

    factor = jnp.sum(
        weighted_currents,
        axis=-1,
    )

    return jnp.moveaxis(
        factor,
        (-2, -1),
        (0, 1),
    )


def jones(
    az_rad: ArrayLike,
    za_rad: ArrayLike,
    dipole_flags: ArrayLike | None = None,
    dipole_gains: ArrayLike | None = None,
) -> Array:
    """
    Evaluate the full 137 MHz MWA tile Jones matrix.

    Parameters
    ----------
    az_rad
        Azimuth in radians, measured eastward from North.
    za_rad
        Zenith angle in radians.
    dipole_flags
        Boolean dipole flags applied before the coupled-current solve.
    dipole_gains
        Real or complex post-current dipole gains applied in the array factor.

    Returns
    -------
    tile_jones
        Complex Jones matrix with shape
        ``(2, 2, *broadcast_shape)``.

        Axis zero contains the sky-vector components ``(phi, theta)``.
        Axis one contains the tile feeds ``(X, Y)``.

    Notes
    -----
    The tile Jones matrix is assembled following the formulation of
    Sutinjo et al. (2014) and the reference implementation in pyuvdata.
    The array-factor matrix computed internally represents the responses
    to independently driven feed excitations; these are combined to form
    the physical per-port array factors before applying them to the
    embedded-element Jones matrix.
    """
    element = element_jones(
        az_rad,
        za_rad,
    )

    factor = array_factor(
        az_rad,
        za_rad,
        dipole_flags=dipole_flags,
        dipole_gains=dipole_gains,
    )

    # factor has shape:
    #     (2 port polarizations, 2 driven feeds, *broadcast_shape)
    #
    # pyuvdata combines the independently driven-feed responses to form one
    # array factor for each port polarization:
    #     (2 port polarizations, *broadcast_shape)
    port_array_factor = jnp.sum(
        factor,
        axis=1,
    )

    # element has shape:
    #     (2 sky-vector components, 2 port polarizations, *broadcast_shape)
    #
    # Apply the corresponding array factor to each Jones column. This
    # deliberately follows pyuvdata's MWA AEE implementation.
    return element * port_array_factor[jnp.newaxis, ...]


def coherency(
    az_rad: ArrayLike,
    za_rad: ArrayLike,
    dipole_flags: ArrayLike | None = None,
    dipole_gains: ArrayLike | None = None,
    *,
    normalize: bool = False,
) -> Array:
    """
    Evaluate the full feed coherency response to an unpolarized source.

    Parameters
    ----------
    az_rad
        Azimuth in radians, measured eastward from North.
    za_rad
        Zenith angle in radians.
    dipole_flags
        Boolean dipole flags applied before the coupled-current solve.
    dipole_gains
        Real or complex post-current dipole gains applied in the array factor.
    normalize
        If True, normalize each coherency term using the corresponding
        zenith feed powers.

    Returns
    -------
    response
        Complex coherency response with shape
        ``(2, 2, *broadcast_shape)``.

        The leading axes contain::

            [[XX, XY],
             [YX, YY]]
    """
    tile_jones = jones(
        az_rad,
        za_rad,
        dipole_flags=dipole_flags,
        dipole_gains=dipole_gains,
    )

    # Convert from:
    #
    #     (sky component, feed, ...)
    #
    # to:
    #
    #     (..., sky component, feed)
    jones_matrix = jnp.moveaxis(
        tile_jones,
        (0, 1),
        (-2, -1),
    )

    # For an unpolarized source, the feed coherency is J^H J.
    jones_hermitian = jnp.swapaxes(
        jones_matrix.conj(),
        -2,
        -1,
    )

    response = jones_hermitian @ jones_matrix

    if normalize:
        zenith_jones = jones(
            jnp.asarray(0.0, dtype=REAL_DTYPE),
            jnp.asarray(0.0, dtype=REAL_DTYPE),
            dipole_flags=dipole_flags,
            dipole_gains=dipole_gains,
        )

        zenith_jones_matrix = jnp.moveaxis(
            zenith_jones,
            (0, 1),
            (-2, -1),
        )

        zenith_response = (
            jnp.swapaxes(
                zenith_jones_matrix.conj(),
                -2,
                -1,
            )
            @ zenith_jones_matrix
        )

        zenith_power = jnp.real(jnp.diag(zenith_response))

        normalization = jnp.sqrt(zenith_power[:, jnp.newaxis] * zenith_power[jnp.newaxis, :])

        response = response / normalization

    return jnp.moveaxis(
        response,
        (-2, -1),
        (0, 1),
    )


def power(
    az_rad: ArrayLike,
    za_rad: ArrayLike,
    dipole_flags: ArrayLike | None = None,
    dipole_gains: ArrayLike | None = None,
    *,
    normalize: bool = False,
) -> Array:
    """
    Evaluate the XX and YY power responses.

    Parameters
    ----------
    az_rad
        Azimuth in radians, measured eastward from North.
    za_rad
        Zenith angle in radians.
    dipole_flags
        Boolean dipole flags applied before the coupled-current solve.
    dipole_gains
        Real or complex post-current dipole gains applied in the array factor.
    normalize
        If True, independently normalize the XX and YY beams by their zenith
        responses.

    Returns
    -------
    response
        Real power response with shape ``(2, *broadcast_shape)``.

        Axis zero contains ``(XX, YY)``.
    """
    response = coherency(
        az_rad,
        za_rad,
        dipole_flags=dipole_flags,
        dipole_gains=dipole_gains,
        normalize=normalize,
    )

    xx = jnp.real(response[0, 0])
    yy = jnp.real(response[1, 1])

    return jnp.stack(
        (xx, yy),
        axis=0,
    )
