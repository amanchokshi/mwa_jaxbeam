import numpy as np

from mwa_jaxbeam import direction_enu


def test_direction_enu_cardinal_directions() -> None:
    az_deg = np.array([0.0, 90.0, 180.0, 270.0, 0.0])
    za_deg = np.array([90.0, 90.0, 90.0, 90.0, 0.0])

    expected = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    directions = np.asarray(
        direction_enu(
            az_rad=np.deg2rad(az_deg),
            za_rad=np.deg2rad(za_deg),
        )
    )

    np.testing.assert_allclose(directions, expected, atol=1e-6)


def test_direction_enu_returns_unit_vectors() -> None:
    az_rad = np.deg2rad(np.array([13.0, 71.0, 145.0, 231.0, 319.0]))
    za_rad = np.deg2rad(np.array([5.0, 24.0, 47.0, 68.0, 89.0]))

    directions = np.asarray(direction_enu(az_rad=az_rad, za_rad=za_rad))

    assert directions.shape == (az_rad.size, 3)
    np.testing.assert_allclose(
        np.linalg.norm(directions, axis=-1),
        1.0,
        atol=1e-6,
    )
