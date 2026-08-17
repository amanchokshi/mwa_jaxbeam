import jax.numpy as jnp
import numpy as np

from mwa_jaxbeam import coherency, power


def test_power_matches_coherency_diagonal() -> None:
    az_rad = jnp.array([0.0, 0.5, 1.0])
    za_rad = jnp.array([0.0, 0.2, 0.4])

    response = coherency(
        az_rad,
        za_rad,
        normalize=False,
    )

    expected = jnp.stack(
        (
            jnp.real(response[0, 0]),
            jnp.real(response[1, 1]),
        ),
        axis=0,
    )

    actual = power(
        az_rad,
        za_rad,
        normalize=False,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_power_output_shape() -> None:
    az_rad = jnp.zeros((3, 4))
    za_rad = jnp.zeros((3, 4))

    response = power(
        az_rad,
        za_rad,
    )

    assert response.shape == (2, 3, 4)


def test_power_is_real_and_nonnegative() -> None:
    az_rad = jnp.linspace(0.0, 2.0 * jnp.pi, 16, endpoint=False)
    za_rad = jnp.linspace(0.0, 0.5 * jnp.pi, 16)

    response = power(
        az_rad,
        za_rad,
        normalize=False,
    )

    assert jnp.issubdtype(response.dtype, jnp.floating)
    assert np.all(np.asarray(response) >= -1e-6)


def test_power_scales_with_common_gain_power() -> None:
    az_rad = jnp.array([0.2, 0.7, 1.4])
    za_rad = jnp.array([0.1, 0.4, 0.8])
    gain = 0.8 * np.exp(1j * np.deg2rad(13.0))

    reference = np.asarray(
        power(
            az_rad,
            za_rad,
            normalize=False,
        )
    )
    scaled = np.asarray(
        power(
            az_rad,
            za_rad,
            dipole_gains=gain,
            normalize=False,
        )
    )

    np.testing.assert_allclose(
        scaled,
        np.abs(gain) ** 2 * reference,
        rtol=1e-5,
        atol=1e-6,
    )


def test_normalized_power_is_unity_at_zenith() -> None:
    response = power(
        0.0,
        0.0,
        normalize=True,
    )

    np.testing.assert_allclose(
        response,
        np.ones(2),
        rtol=1e-5,
        atol=1e-6,
    )


def test_common_gain_cancels_in_normalized_power() -> None:
    az_rad = jnp.array([0.2, 0.7, 1.4])
    za_rad = jnp.array([0.1, 0.4, 0.8])
    gain = 0.8 * np.exp(1j * np.deg2rad(13.0))

    reference = np.asarray(
        power(
            az_rad,
            za_rad,
            normalize=True,
        )
    )
    scaled = np.asarray(
        power(
            az_rad,
            za_rad,
            dipole_gains=gain,
            normalize=True,
        )
    )

    np.testing.assert_allclose(
        scaled,
        reference,
        rtol=1e-5,
        atol=1e-6,
    )
