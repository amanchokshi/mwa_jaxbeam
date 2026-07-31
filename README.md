# mwa_jaxbeam

`mwa_jaxbeam` is a pure JAX implementation of the Murchison Widefield Array (MWA) Average Embedded Element (AEE) beam model.

The package reproduces the 137 MHz AEE model distributed with `pyuvdata` while providing a differentiable implementation suitable for optimization and inference problems. It is designed for applications such as satellite-based beam calibration, where the beam model must be evaluated millions of times and differentiated with respect to instrument parameters.

> [!WARNING]
> This is **experimental software**. At present, only the **137 MHz** MWA AEE beam model with **zenith pointing** is implemented.

## Features

- Pure JAX implementation with automatic differentiation.
- Numerically validated against the MWA AEE implementation in `pyuvdata`.
- Fast evaluation of the full complex tile Jones matrix.
- Support for arbitrary real or complex dipole excitations.
- Efficient JIT compilation for large sky grids.

## Installation

```bash
git clone https://github.com/amanchokshi/mwa_jaxbeam.git
cd mwa_jaxbeam

uv sync
```

or

```bash
pip install -e .
```

## Example

```python
import jax.numpy as jnp

from mwa_jaxbeam.aee_137mhz import jones

az_rad = jnp.deg2rad(jnp.linspace(0.0, 360.0, 361))

za_rad = jnp.deg2rad(jnp.linspace(0.0, 90.0, 91))

az_grid, za_grid = jnp.meshgrid(
    az_rad,
    za_rad,
)

beam = jones(
    az_rad=az_grid,
    za_rad=za_grid,
)
```

The returned Jones matrix has shape

```text
(2, 2, *broadcast_shape)
```

where the first axis contains the sky-vector components `(φ, θ)` and the second axis contains the tile feeds `(X, Y)`.

## Validation

The implementation has been validated against the MWA AEE model distributed with `pyuvdata` by independently reproducing each stage of the calculation

## Performance

On an Apple M4 Pro, evaluation of the full tile Jones matrix for a typical satellite beam-calibration workload (6144 sky directions) takes approximately

- **0.5 ms** per forward evaluation (~2000 evaluations/s),
- **0.86 ms** for a forward evaluation plus reverse-mode gradient (~1160 evaluations/s).

Performance scales approximately linearly with the number of sky directions after JIT compilation.

## Status

The current implementation supports

- the 137 MHz MWA Average Embedded Element beam model,
- arbitrary complex dipole excitations,
- automatic differentiation through the complete beam calculation.

Future work includes interpolation across frequency and support for arbitrary beamformer delay settings.

## Citation

If you use this package in published work, please cite the relevant MWA beam-model and `pyuvdata` publications, together with this repository.
