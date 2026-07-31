import numpy as np

from mwa_jaxbeam import (
    DIPOLE_POSITIONS_ENU_M,
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

    grid = np.asarray(array_factor(az_rad=az[np.newaxis, :], za_rad=za[:, np.newaxis]))

    for za_index, za_value in enumerate(za):
        for az_index, az_value in enumerate(az):
            scalar = np.asarray(array_factor(az_rad=az_value, za_rad=za_value))
            np.testing.assert_allclose(
                grid[:, :, za_index, az_index],
                scalar,
                rtol=1e-6,
                atol=1e-7,
            )


def test_array_factor_zenith_equals_summed_currents() -> None:
    excitations = np.ones((2, 16), dtype=np.complex64)
    currents = np.asarray(port_currents(excitations=excitations))
    factor = np.asarray(array_factor(az_rad=0.0, za_rad=0.0, excitations=excitations))

    np.testing.assert_allclose(factor, np.sum(currents, axis=-1), rtol=1e-6, atol=1e-7)


def test_array_factor_zenith_is_azimuth_independent() -> None:
    excitations = np.ones((2, 16), dtype=np.complex64)
    azimuths = np.deg2rad(np.array([0.0, 45.0, 90.0, 180.0, 270.0, 359.0]))

    factor = np.asarray(array_factor(az_rad=azimuths, za_rad=0.0, excitations=excitations))
    expected = np.asarray(array_factor(az_rad=0.0, za_rad=0.0, excitations=excitations))
    expected = np.broadcast_to(expected[:, :, np.newaxis], factor.shape)

    np.testing.assert_allclose(factor, expected, rtol=1e-6, atol=1e-7)


def test_array_factor_scalar_independent_reconstruction() -> None:
    az_rad = np.deg2rad(63.0)
    za_rad = np.deg2rad(37.0)
    excitations = np.ones((2, 16), dtype=np.complex64)

    currents = np.asarray(port_currents(excitations=excitations))
    direction = np.asarray(direction_enu(az_rad=az_rad, za_rad=za_rad))
    path = np.sum(np.asarray(DIPOLE_POSITIONS_ENU_M) * direction, axis=-1)
    phase = np.exp(1j * WAVENUMBER_RAD_PER_M * path)
    expected = np.sum(currents * phase[np.newaxis, np.newaxis, :], axis=-1)

    actual = np.asarray(array_factor(az_rad=az_rad, za_rad=za_rad, excitations=excitations))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.abs(phase), 1.0, rtol=1e-6, atol=1e-7)


def test_array_factor_grid_independent_reconstruction() -> None:
    az = np.deg2rad(np.array([0.0, 45.0, 120.0, 270.0]))
    za = np.deg2rad(np.array([0.0, 20.0, 50.0]))
    az_grid = az[np.newaxis, :]
    za_grid = za[:, np.newaxis]
    excitations = np.ones((2, 16), dtype=np.complex64)

    currents = np.asarray(port_currents(excitations=excitations))
    direction = np.asarray(direction_enu(az_rad=az_grid, za_rad=za_grid))
    path = np.sum(
        direction[..., np.newaxis, :] * np.asarray(DIPOLE_POSITIONS_ENU_M),
        axis=-1,
    )
    phase = np.exp(1j * WAVENUMBER_RAD_PER_M * path)
    expected = np.sum(
        currents[:, :, np.newaxis, np.newaxis, :] * phase[np.newaxis, np.newaxis, ...],
        axis=-1,
    )

    actual = np.asarray(array_factor(az_rad=az_grid, za_rad=za_grid, excitations=excitations))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_array_factor_is_complex_linear() -> None:
    az = np.deg2rad(np.array([15.0, 90.0, 210.0]))
    za = np.deg2rad(np.array([10.0, 35.0, 60.0]))
    scale = 2.0 - 0.5j

    reference = np.asarray(array_factor(az_rad=az, za_rad=za, excitations=1.0))
    scaled = np.asarray(array_factor(az_rad=az, za_rad=za, excitations=scale))

    np.testing.assert_allclose(scaled, scale * reference, rtol=1e-5, atol=1e-6)


def test_zero_excitation_produces_zero_array_factor() -> None:
    az = np.deg2rad(np.linspace(0.0, 360.0, 13, endpoint=False))
    za = np.deg2rad(np.linspace(0.0, 90.0, 7))

    factor = np.asarray(
        array_factor(
            az_rad=az[np.newaxis, :],
            za_rad=za[:, np.newaxis],
            excitations=0.0,
        )
    )

    np.testing.assert_allclose(factor, 0.0, atol=1e-7)


def test_array_factor_north_south_power_symmetry() -> None:
    za = np.deg2rad(np.linspace(0.0, 90.0, 181))

    north = np.asarray(array_factor(az_rad=0.0, za_rad=za, excitations=1.0))
    south = np.asarray(array_factor(az_rad=np.pi, za_rad=za, excitations=1.0))

    np.testing.assert_allclose(
        np.abs(north) ** 2,
        np.abs(south) ** 2,
        rtol=1e-5,
        atol=1e-7,
    )
