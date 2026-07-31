import numpy as np

from mwa_jaxbeam import (
    AZIMUTH_RAD,
    DIPOLE_POSITIONS_ENU_M,
    ELEMENT_JONES,
    N_DIPOLES,
    N_FEEDS,
    N_PORTS,
    Z_TOTAL_OHM,
    ZA_RAD,
)


def test_packaged_runtime_data_integrity() -> None:
    az = np.asarray(AZIMUTH_RAD)
    za = np.asarray(ZA_RAD)
    element_jones = np.asarray(ELEMENT_JONES)
    z_total = np.asarray(Z_TOTAL_OHM)
    positions = np.asarray(DIPOLE_POSITIONS_ENU_M)

    assert element_jones.shape == (N_FEEDS, N_FEEDS, za.size, az.size)
    assert z_total.shape == (N_PORTS, N_PORTS)
    assert positions.shape == (N_DIPOLES, 3)

    assert element_jones.dtype == np.complex64
    assert z_total.dtype == np.complex64
    assert positions.dtype == np.float32
    assert az.dtype == np.float32
    assert za.dtype == np.float32

    assert np.all(np.isfinite(element_jones))
    assert np.all(np.isfinite(z_total))
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(az))
    assert np.all(np.isfinite(za))

    assert np.all(np.diff(az) > 0)
    assert np.all(np.diff(za) > 0)

    assert np.isclose(az[0], 0.0)
    assert np.isclose(np.rad2deg(az[-1]), 357.0)
    assert np.isclose(za[0], 0.0)
    assert np.isclose(np.rad2deg(za[-1]), 90.0)

    assert np.allclose(positions[:, 2], positions[0, 2])
    assert np.unique(positions[:, 0]).size == 4
    assert np.unique(positions[:, 1]).size == 4
    assert np.allclose(np.diff(np.unique(positions[:, 0])), 1.1)
    assert np.allclose(np.diff(np.unique(positions[:, 1])), 1.1)
