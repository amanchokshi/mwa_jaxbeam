import numpy as np

from mwa_jaxbeam import coherency, jones


def test_coherency_shape_and_independent_reconstruction() -> None:
    az = np.deg2rad(np.array([15.0, 90.0, 210.0]))
    za = np.deg2rad(np.array([10.0, 35.0, 60.0]))

    response = np.asarray(coherency(az_rad=az, za_rad=za))
    assert response.shape == (2, 2, 3)
    assert np.iscomplexobj(response)
    assert np.all(np.isfinite(response))

    tile_jones = np.asarray(jones(az_rad=az, za_rad=za))
    jones_matrix = np.moveaxis(tile_jones, (0, 1), (-2, -1))
    expected = np.swapaxes(jones_matrix.conj(), -2, -1) @ jones_matrix
    expected = np.moveaxis(expected, (-2, -1), (0, 1))

    np.testing.assert_allclose(response, expected, rtol=1e-6, atol=1e-7)


def test_coherency_is_hermitian_positive_semidefinite() -> None:
    az = np.deg2rad(np.array([15.0, 90.0, 210.0]))
    za = np.deg2rad(np.array([10.0, 35.0, 60.0]))
    response = np.asarray(coherency(az_rad=az, za_rad=za))

    np.testing.assert_allclose(response[1, 0], response[0, 1].conj(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(response[0, 0].imag, 0.0, atol=1e-7)
    np.testing.assert_allclose(response[1, 1].imag, 0.0, atol=1e-7)

    assert np.all(response[0, 0].real >= -1e-7)
    assert np.all(response[1, 1].real >= -1e-7)

    response_matrix = np.moveaxis(response, (0, 1), (-2, -1))
    assert np.all(np.linalg.eigvalsh(response_matrix) >= -1e-6)


def test_normalized_zenith_coherency_has_unit_diagonal() -> None:
    response = np.asarray(coherency(az_rad=0.0, za_rad=0.0, normalize=True))

    np.testing.assert_allclose(response[0, 0], 1.0, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(response[1, 1], 1.0, rtol=1e-6, atol=1e-7)


def test_coherency_scales_with_excitation_power() -> None:
    az = np.deg2rad(np.array([15.0, 90.0, 210.0]))
    za = np.deg2rad(np.array([10.0, 35.0, 60.0]))
    scale = 2.0 - 0.5j

    reference = np.asarray(coherency(az_rad=az, za_rad=za, excitations=1.0))
    scaled = np.asarray(coherency(az_rad=az, za_rad=za, excitations=scale))

    np.testing.assert_allclose(
        scaled,
        np.abs(scale) ** 2 * reference,
        rtol=1e-5,
        atol=1e-6,
    )


def test_zero_excitation_produces_zero_coherency() -> None:
    az = np.deg2rad(np.array([15.0, 90.0, 210.0]))
    za = np.deg2rad(np.array([10.0, 35.0, 60.0]))

    response = np.asarray(coherency(az_rad=az, za_rad=za, excitations=0.0))
    np.testing.assert_allclose(response, 0.0, atol=1e-7)
