import jax.numpy as jnp
import numpy as np

from mwa_jaxbeam import (
    AZIMUTH_RAD,
    ELEMENT_JONES,
    N_FEEDS,
    ZA_RAD,
    element_jones,
)


def test_element_jones_reproduces_all_grid_nodes() -> None:
    az_grid, za_grid = jnp.meshgrid(
        AZIMUTH_RAD,
        ZA_RAD,
        indexing="xy",
    )

    interpolated = np.asarray(element_jones(az_rad=az_grid, za_rad=za_grid))
    stored = np.asarray(ELEMENT_JONES)

    assert interpolated.shape == stored.shape
    np.testing.assert_allclose(interpolated, stored, rtol=1e-6, atol=1e-6)


def test_element_jones_scalar_grid_node() -> None:
    za_index = 10
    az_index = 20

    interpolated = np.asarray(
        element_jones(
            az_rad=AZIMUTH_RAD[az_index],
            za_rad=ZA_RAD[za_index],
        )
    )
    stored = np.asarray(ELEMENT_JONES)[:, :, za_index, az_index]

    assert interpolated.shape == (N_FEEDS, N_FEEDS)
    np.testing.assert_allclose(interpolated, stored, rtol=1e-6, atol=1e-6)


def test_element_jones_bilinear_midpoint() -> None:
    za_index = 10
    az_index = 20
    stored = np.asarray(ELEMENT_JONES)

    za_mid = 0.5 * (ZA_RAD[za_index] + ZA_RAD[za_index + 1])
    az_mid = 0.5 * (AZIMUTH_RAD[az_index] + AZIMUTH_RAD[az_index + 1])

    interpolated = np.asarray(element_jones(az_rad=az_mid, za_rad=za_mid))
    expected = 0.25 * (
        stored[:, :, za_index, az_index]
        + stored[:, :, za_index + 1, az_index]
        + stored[:, :, za_index, az_index + 1]
        + stored[:, :, za_index + 1, az_index + 1]
    )

    np.testing.assert_allclose(interpolated, expected, rtol=1e-5, atol=1e-6)


def test_element_jones_wraps_two_pi_to_zero() -> None:
    za_rad = np.deg2rad(36.0)

    at_zero = np.asarray(element_jones(az_rad=0.0, za_rad=za_rad))
    at_two_pi = np.asarray(element_jones(az_rad=2.0 * np.pi, za_rad=za_rad))

    np.testing.assert_allclose(at_zero, at_two_pi, rtol=1e-6, atol=1e-6)


def test_element_jones_wraps_negative_azimuth() -> None:
    za_rad = np.deg2rad(36.0)

    negative = np.asarray(element_jones(az_rad=np.deg2rad(-3.0), za_rad=za_rad))
    positive = np.asarray(element_jones(az_rad=np.deg2rad(357.0), za_rad=za_rad))

    np.testing.assert_allclose(negative, positive, rtol=1e-6, atol=1e-6)
