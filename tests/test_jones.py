import numpy as np

from mwa_jaxbeam import array_factor, element_jones, jones, port_currents


def test_jones_output_shapes_and_finiteness() -> None:
    scalar = np.asarray(jones(az_rad=0.0, za_rad=0.0))
    assert scalar.shape == (2, 2)
    assert np.iscomplexobj(scalar)
    assert np.all(np.isfinite(scalar))

    az_vector = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0]))
    za_vector = np.deg2rad(np.array([10.0, 20.0, 30.0, 40.0]))
    vector = np.asarray(jones(az_rad=az_vector, za_rad=za_vector))
    assert vector.shape == (2, 2, 4)
    assert np.iscomplexobj(vector)
    assert np.all(np.isfinite(vector))

    az_grid = np.deg2rad(np.linspace(0.0, 360.0, 13, endpoint=False))
    za_grid = np.deg2rad(np.linspace(0.0, 90.0, 7))
    grid = np.asarray(
        jones(az_rad=az_grid[np.newaxis, :], za_rad=za_grid[:, np.newaxis])
    )
    assert grid.shape == (2, 2, za_grid.size, az_grid.size)
    assert np.iscomplexobj(grid)
    assert np.all(np.isfinite(grid))


def test_jones_vectorized_and_scalar_evaluations_agree() -> None:
    az = np.deg2rad(np.linspace(0.0, 360.0, 13, endpoint=False))
    za = np.deg2rad(np.linspace(0.0, 90.0, 7))
    grid = np.asarray(jones(az_rad=az[np.newaxis, :], za_rad=za[:, np.newaxis]))

    for za_index, za_value in enumerate(za):
        for az_index, az_value in enumerate(az):
            scalar = np.asarray(jones(az_rad=az_value, za_rad=za_value))
            np.testing.assert_allclose(
                grid[:, :, za_index, az_index],
                scalar,
                rtol=1e-6,
                atol=1e-7,
            )


def test_jones_scalar_pyuvdata_compatible_reconstruction() -> None:
    az_rad = np.deg2rad(63.0)
    za_rad = np.deg2rad(37.0)
    excitations = np.ones((2, 16), dtype=np.complex64)

    element = np.asarray(element_jones(az_rad=az_rad, za_rad=za_rad))
    factor = np.asarray(
        array_factor(az_rad=az_rad, za_rad=za_rad, excitations=excitations)
    )
    expected = element * np.sum(factor, axis=1)[np.newaxis, :]
    actual = np.asarray(jones(az_rad=az_rad, za_rad=za_rad, excitations=excitations))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_jones_grid_pyuvdata_compatible_reconstruction() -> None:
    az = np.deg2rad(np.array([0.0, 45.0, 120.0, 270.0]))
    za = np.deg2rad(np.array([0.0, 20.0, 50.0]))
    az_grid = az[np.newaxis, :]
    za_grid = za[:, np.newaxis]
    excitations = np.ones((2, 16), dtype=np.complex64)

    element = np.asarray(element_jones(az_rad=az_grid, za_rad=za_grid))
    factor = np.asarray(
        array_factor(az_rad=az_grid, za_rad=za_grid, excitations=excitations)
    )
    expected = element * np.sum(factor, axis=1)[np.newaxis, ...]
    actual = np.asarray(jones(az_rad=az_grid, za_rad=za_grid, excitations=excitations))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_jones_zenith_analytic_reconstruction() -> None:
    excitations = np.ones((2, 16), dtype=np.complex64)
    element = np.asarray(element_jones(az_rad=0.0, za_rad=0.0))
    currents = np.asarray(port_currents(excitations=excitations))

    port_array_factor = np.sum(np.sum(currents, axis=-1), axis=1)
    expected = element * port_array_factor[np.newaxis, :]
    actual = np.asarray(jones(az_rad=0.0, za_rad=0.0, excitations=excitations))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_jones_is_complex_linear() -> None:
    az = np.deg2rad(np.array([15.0, 90.0, 210.0]))
    za = np.deg2rad(np.array([10.0, 35.0, 60.0]))
    scale = 2.0 - 0.5j

    reference = np.asarray(jones(az_rad=az, za_rad=za, excitations=1.0))
    scaled = np.asarray(jones(az_rad=az, za_rad=za, excitations=scale))

    np.testing.assert_allclose(scaled, scale * reference, rtol=1e-5, atol=1e-6)


def test_zero_excitation_produces_zero_jones() -> None:
    az = np.deg2rad(np.linspace(0.0, 360.0, 13, endpoint=False))
    za = np.deg2rad(np.linspace(0.0, 90.0, 7))

    result = np.asarray(
        jones(
            az_rad=az[np.newaxis, :],
            za_rad=za[:, np.newaxis],
            excitations=0.0,
        )
    )

    np.testing.assert_allclose(result, 0.0, atol=1e-7)
