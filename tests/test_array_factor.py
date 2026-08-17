import numpy as np
import pytest

from mwa_jaxbeam import (
    DIPOLE_POSITIONS_ENU_M,
    N_DIPOLES,
    N_FEEDS,
    WAVENUMBER_RAD_PER_M,
    array_factor,
    direction_enu,
    port_currents,
)


def test_array_factor_output_shapes_and_finiteness() -> None:
    scalar = np.asarray(array_factor(az_rad=0.0, za_rad=0.0))
    assert scalar.shape == (2, 2)
    assert np.iscomplexobj(scalar)
    assert np.all(np.isfinite(scalar))

    az_vector = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0]))
    za_vector = np.deg2rad(np.array([10.0, 20.0, 30.0, 40.0]))
    vector = np.asarray(array_factor(az_rad=az_vector, za_rad=za_vector))
    assert vector.shape == (2, 2, 4)
    assert np.iscomplexobj(vector)
    assert np.all(np.isfinite(vector))

    az_grid = np.deg2rad(np.linspace(0.0, 360.0, 13, endpoint=False))
    za_grid = np.deg2rad(np.linspace(0.0, 90.0, 7))
    grid = np.asarray(
        array_factor(
            az_rad=az_grid[np.newaxis, :],
            za_rad=za_grid[:, np.newaxis],
        )
    )
    assert grid.shape == (2, 2, za_grid.size, az_grid.size)
    assert np.iscomplexobj(grid)
    assert np.all(np.isfinite(grid))


def test_array_factor_vectorized_and_scalar_evaluations_agree() -> None:
    az = np.deg2rad(np.linspace(0.0, 360.0, 13, endpoint=False))
    za = np.deg2rad(np.linspace(0.0, 90.0, 7))

    grid = np.asarray(
        array_factor(
            az_rad=az[np.newaxis, :],
            za_rad=za[:, np.newaxis],
        )
    )

    for za_index, za_value in enumerate(za):
        for az_index, az_value in enumerate(az):
            scalar = np.asarray(
                array_factor(
                    az_rad=az_value,
                    za_rad=za_value,
                )
            )
            np.testing.assert_allclose(
                grid[:, :, za_index, az_index],
                scalar,
                rtol=1e-6,
                atol=1e-7,
            )


def test_accepted_gain_representations_are_equivalent_at_unity() -> None:
    default = np.asarray(array_factor(az_rad=0.7, za_rad=0.4))
    scalar = np.asarray(
        array_factor(
            az_rad=0.7,
            za_rad=0.4,
            dipole_gains=1.0,
        )
    )
    per_dipole = np.asarray(
        array_factor(
            az_rad=0.7,
            za_rad=0.4,
            dipole_gains=np.ones(N_DIPOLES, dtype=np.complex64),
        )
    )
    per_feed_dipole = np.asarray(
        array_factor(
            az_rad=0.7,
            za_rad=0.4,
            dipole_gains=np.ones((N_FEEDS, N_DIPOLES), dtype=np.complex64),
        )
    )

    np.testing.assert_allclose(default, scalar, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(default, per_dipole, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(default, per_feed_dipole, rtol=1e-6, atol=1e-7)


def test_array_factor_zenith_equals_gain_weighted_summed_currents() -> None:
    gains = np.ones((N_FEEDS, N_DIPOLES), dtype=np.complex64)
    gains[0, 3] = 0.8 * np.exp(1j * np.deg2rad(12.0))
    gains[1, 9] = 1.1 * np.exp(-1j * np.deg2rad(7.0))

    currents = np.asarray(port_currents())
    expected = np.sum(currents * gains[np.newaxis, :, :], axis=-1)
    factor = np.asarray(
        array_factor(
            az_rad=0.0,
            za_rad=0.0,
            dipole_gains=gains,
        )
    )

    np.testing.assert_allclose(factor, expected, rtol=1e-6, atol=1e-7)


def test_array_factor_zenith_is_azimuth_independent() -> None:
    azimuths = np.deg2rad(np.array([0.0, 45.0, 90.0, 180.0, 270.0, 359.0]))

    factor = np.asarray(
        array_factor(
            az_rad=azimuths,
            za_rad=0.0,
        )
    )
    expected = np.asarray(array_factor(az_rad=0.0, za_rad=0.0))
    expected = np.broadcast_to(expected[:, :, np.newaxis], factor.shape)

    np.testing.assert_allclose(factor, expected, rtol=1e-6, atol=1e-7)


def test_array_factor_scalar_independent_reconstruction() -> None:
    az_rad = np.deg2rad(63.0)
    za_rad = np.deg2rad(37.0)

    flags = np.ones((N_FEEDS, N_DIPOLES), dtype=bool)
    flags[0, 5] = False

    gains = np.ones((N_FEEDS, N_DIPOLES), dtype=np.complex64)
    gains[0, 2] = 0.7 * np.exp(1j * np.deg2rad(11.0))
    gains[1, 12] = 1.2 * np.exp(-1j * np.deg2rad(4.0))

    currents = np.asarray(port_currents(dipole_flags=flags))
    direction = np.asarray(direction_enu(az_rad=az_rad, za_rad=za_rad))
    path = np.sum(
        np.asarray(DIPOLE_POSITIONS_ENU_M) * direction,
        axis=-1,
    )
    phase = np.exp(1j * WAVENUMBER_RAD_PER_M * path)

    expected = np.sum(
        currents * gains[np.newaxis, :, :] * phase[np.newaxis, np.newaxis, :],
        axis=-1,
    )
    actual = np.asarray(
        array_factor(
            az_rad=az_rad,
            za_rad=za_rad,
            dipole_flags=flags,
            dipole_gains=gains,
        )
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.abs(phase), 1.0, rtol=1e-6, atol=1e-7)


def test_array_factor_grid_independent_reconstruction() -> None:
    az = np.deg2rad(np.array([0.0, 45.0, 120.0, 270.0]))
    za = np.deg2rad(np.array([0.0, 20.0, 50.0]))
    az_grid = az[np.newaxis, :]
    za_grid = za[:, np.newaxis]

    gains = np.ones((N_FEEDS, N_DIPOLES), dtype=np.complex64)
    gains[0] *= 0.9 * np.exp(1j * np.deg2rad(3.0))
    gains[1] *= 1.1 * np.exp(-1j * np.deg2rad(5.0))

    currents = np.asarray(port_currents())
    direction = np.asarray(direction_enu(az_rad=az_grid, za_rad=za_grid))
    path = np.sum(
        direction[..., np.newaxis, :] * np.asarray(DIPOLE_POSITIONS_ENU_M),
        axis=-1,
    )
    phase = np.exp(1j * WAVENUMBER_RAD_PER_M * path)
    expected = np.sum(
        (currents * gains[np.newaxis, :, :])[:, :, np.newaxis, np.newaxis, :] * phase[np.newaxis, np.newaxis, ...],
        axis=-1,
    )

    actual = np.asarray(
        array_factor(
            az_rad=az_grid,
            za_rad=za_grid,
            dipole_gains=gains,
        )
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_common_complex_gain_scales_array_factor() -> None:
    az = np.deg2rad(np.array([15.0, 90.0, 210.0]))
    za = np.deg2rad(np.array([10.0, 35.0, 60.0]))
    gain = 2.0 - 0.5j

    reference = np.asarray(array_factor(az_rad=az, za_rad=za))
    scaled = np.asarray(
        array_factor(
            az_rad=az,
            za_rad=za,
            dipole_gains=gain,
        )
    )

    np.testing.assert_allclose(scaled, gain * reference, rtol=1e-5, atol=1e-6)


def test_zero_gains_produce_zero_array_factor() -> None:
    az = np.deg2rad(np.linspace(0.0, 360.0, 13, endpoint=False))
    za = np.deg2rad(np.linspace(0.0, 90.0, 7))

    factor = np.asarray(
        array_factor(
            az_rad=az[np.newaxis, :],
            za_rad=za[:, np.newaxis],
            dipole_gains=0.0,
        )
    )

    np.testing.assert_allclose(factor, 0.0, atol=1e-7)


def test_gains_do_not_change_port_currents() -> None:
    currents_before = np.asarray(port_currents())

    gains = np.ones((N_FEEDS, N_DIPOLES), dtype=np.complex64)
    gains[0, 4] = 0.3 + 0.7j
    _ = array_factor(
        az_rad=0.5,
        za_rad=0.3,
        dipole_gains=gains,
    )

    currents_after = np.asarray(port_currents())
    np.testing.assert_allclose(currents_after, currents_before, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "dipole_gains",
    [
        np.ones(2 * N_DIPOLES),
        np.ones((N_DIPOLES, N_FEEDS)),
        np.ones((1, N_DIPOLES)),
        np.ones((N_FEEDS, N_DIPOLES, 1)),
    ],
)
def test_invalid_gain_shapes_raise(dipole_gains: np.ndarray) -> None:
    with pytest.raises(ValueError):
        array_factor(
            az_rad=0.0,
            za_rad=0.0,
            dipole_gains=dipole_gains,
        )


def test_array_factor_north_south_power_symmetry() -> None:
    za = np.deg2rad(np.linspace(0.0, 90.0, 181))

    north = np.asarray(array_factor(az_rad=0.0, za_rad=za))
    south = np.asarray(array_factor(az_rad=np.pi, za_rad=za))

    np.testing.assert_allclose(
        np.abs(north) ** 2,
        np.abs(south) ** 2,
        rtol=1e-5,
        atol=1e-7,
    )
