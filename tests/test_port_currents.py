import numpy as np
import pytest

from mwa_jaxbeam import (
    N_DIPOLES,
    N_FEEDS,
    N_PORTS,
    Z_TOTAL_OHM,
    port_currents,
)


def test_accepted_flag_representations_are_equivalent() -> None:
    default = np.asarray(port_currents())
    scalar = np.asarray(port_currents(dipole_flags=True))
    per_dipole = np.asarray(
        port_currents(
            dipole_flags=np.ones(N_DIPOLES, dtype=bool),
        )
    )
    per_feed_dipole = np.asarray(
        port_currents(
            dipole_flags=np.ones((N_FEEDS, N_DIPOLES), dtype=bool),
        )
    )

    expected_shape = (N_FEEDS, N_FEEDS, N_DIPOLES)
    for currents in (default, scalar, per_dipole, per_feed_dipole):
        assert currents.shape == expected_shape
        assert currents.dtype == np.complex64

    np.testing.assert_allclose(default, scalar, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(default, per_dipole, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(default, per_feed_dipole, rtol=1e-6, atol=1e-6)


def test_port_currents_are_finite_and_have_expected_slices() -> None:
    currents = np.asarray(port_currents())

    assert currents[0].shape == (N_FEEDS, N_DIPOLES)
    assert currents[1].shape == (N_FEEDS, N_DIPOLES)
    assert np.all(np.isfinite(currents))


def test_port_currents_satisfy_impedance_system_with_flags() -> None:
    flags = np.ones((N_FEEDS, N_DIPOLES), dtype=bool)
    flags[0, 3] = False
    flags[1, 11] = False

    currents = np.asarray(port_currents(dipole_flags=flags))
    z_total = np.asarray(Z_TOTAL_OHM)

    for driven_feed in range(N_FEEDS):
        voltage = np.zeros((N_FEEDS, N_DIPOLES), dtype=np.complex64)
        voltage[driven_feed] = flags[driven_feed].astype(np.complex64)

        reconstructed = z_total @ currents[:, driven_feed, :].reshape(N_PORTS)

        np.testing.assert_allclose(
            reconstructed,
            voltage.reshape(N_PORTS),
            rtol=1e-5,
            atol=1e-6,
        )


def test_feed_dependent_flag_affects_only_corresponding_solution() -> None:
    uniform = np.ones((N_FEEDS, N_DIPOLES), dtype=bool)
    modified = uniform.copy()
    modified[0, 0] = False

    uniform_currents = np.asarray(port_currents(dipole_flags=uniform))
    modified_currents = np.asarray(port_currents(dipole_flags=modified))

    assert not np.allclose(
        modified_currents[:, 0, :],
        uniform_currents[:, 0, :],
    )
    np.testing.assert_allclose(
        modified_currents[:, 1, :],
        uniform_currents[:, 1, :],
        rtol=1e-6,
        atol=1e-6,
    )


def test_all_false_flags_produce_zero_currents() -> None:
    currents = np.asarray(port_currents(dipole_flags=False))
    np.testing.assert_allclose(currents, 0.0, atol=1e-7)


@pytest.mark.parametrize(
    "dipole_flags",
    [
        np.ones(N_PORTS, dtype=bool),
        np.ones((N_DIPOLES, N_FEEDS), dtype=bool),
        np.ones((1, N_DIPOLES), dtype=bool),
        np.ones((N_FEEDS, N_DIPOLES, 1), dtype=bool),
    ],
)
def test_invalid_flag_shapes_raise(dipole_flags: np.ndarray) -> None:
    with pytest.raises(ValueError):
        port_currents(dipole_flags=dipole_flags)


@pytest.mark.parametrize(
    "dipole_flags",
    [
        1,
        np.ones(N_DIPOLES, dtype=np.int32),
        np.ones((N_FEEDS, N_DIPOLES), dtype=np.float32),
    ],
)
def test_non_boolean_flags_raise(dipole_flags: object) -> None:
    with pytest.raises(ValueError, match="boolean"):
        port_currents(dipole_flags=dipole_flags)
