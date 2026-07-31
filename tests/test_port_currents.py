import numpy as np
import pytest

from mwa_jaxbeam import (
    N_DIPOLES,
    N_FEEDS,
    N_PORTS,
    Z_TOTAL_OHM,
    port_currents,
)


def test_accepted_excitation_representations_are_equivalent() -> None:
    scalar = np.asarray(port_currents(excitations=1.0))
    per_dipole = np.asarray(
        port_currents(excitations=np.ones(N_DIPOLES, dtype=np.complex64))
    )
    per_feed_dipole = np.asarray(
        port_currents(excitations=np.ones((N_FEEDS, N_DIPOLES), dtype=np.complex64))
    )

    expected_shape = (N_FEEDS, N_FEEDS, N_DIPOLES)
    for currents in (scalar, per_dipole, per_feed_dipole):
        assert currents.shape == expected_shape
        assert currents.dtype == np.complex64

    np.testing.assert_allclose(scalar, per_dipole, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scalar, per_feed_dipole, rtol=1e-6, atol=1e-6)


def test_port_currents_are_finite_and_have_expected_slices() -> None:
    currents = np.asarray(port_currents(excitations=1.0))

    assert currents[0].shape == (N_FEEDS, N_DIPOLES)
    assert currents[1].shape == (N_FEEDS, N_DIPOLES)
    assert np.all(np.isfinite(currents))


def test_port_currents_satisfy_impedance_system() -> None:
    excitation = np.ones((N_FEEDS, N_DIPOLES), dtype=np.complex64)
    currents = np.asarray(port_currents(excitations=excitation))
    z_total = np.asarray(Z_TOTAL_OHM)

    for driven_feed in range(N_FEEDS):
        voltage = np.zeros((N_FEEDS, N_DIPOLES), dtype=np.complex64)
        voltage[driven_feed] = excitation[driven_feed]

        reconstructed = z_total @ currents[:, driven_feed, :].reshape(N_PORTS)

        np.testing.assert_allclose(
            reconstructed,
            voltage.reshape(N_PORTS),
            rtol=1e-5,
            atol=1e-6,
        )


def test_port_currents_are_complex_linear() -> None:
    scale = 2.0 - 0.5j
    reference = np.asarray(port_currents(excitations=1.0))
    scaled = np.asarray(port_currents(excitations=scale))

    np.testing.assert_allclose(scaled, scale * reference, rtol=1e-5, atol=1e-6)


def test_feed_dependent_excitations_affect_only_corresponding_solution() -> None:
    uniform = np.ones((N_FEEDS, N_DIPOLES), dtype=np.complex64)
    modified = uniform.copy()
    modified[0, 0] = 0.0

    uniform_currents = np.asarray(port_currents(excitations=uniform))
    modified_currents = np.asarray(port_currents(excitations=modified))

    assert not np.allclose(modified_currents[:, 0, :], uniform_currents[:, 0, :])
    np.testing.assert_allclose(
        modified_currents[:, 1, :],
        uniform_currents[:, 1, :],
        rtol=1e-6,
        atol=1e-6,
    )


def test_zero_excitation_produces_zero_currents() -> None:
    currents = np.asarray(port_currents(excitations=0.0))
    np.testing.assert_allclose(currents, 0.0, atol=1e-7)


@pytest.mark.parametrize(
    "excitations",
    [
        np.ones(N_PORTS),
        np.ones((N_DIPOLES, N_FEEDS)),
        np.ones((1, N_DIPOLES)),
        np.ones((N_FEEDS, N_DIPOLES, 1)),
    ],
)
def test_invalid_excitation_shapes_raise(excitations: np.ndarray) -> None:
    with pytest.raises(ValueError):
        port_currents(excitations=excitations)
