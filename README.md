# Discontinuous Diffeomorphic Image Registration with Groupoid

The source code of "A Diffeomorphism Groupoid and Algebroid Framework for Discontinuous Image Registration", which aims to solve the discontinuous sliding motion registration using a diffeomorphism groupoid and algebroid approach, implemented on top of the [mermaid](https://github.com/uncbiag/mermaid) registration library and extended with a FEniCS-based discontinuous velocity solver.

## What This Repository Adds

Compared with upstream `mermaid`, this repository contains:

- a discontinuous diffeomorphism groupoid based registration method for sliding motion
- a FEniCS-based solver for the velocity-momentum relation
- demo scripts for 2D examples
- visualization scripts for deformation grids, quiver plots, and difference maps

## Main Repository Layout

- [`sliding_demo/`](sliding_demo): main demo script
- [`mermaid/`](mermaid): registration framework 
- [`data/`](data): example images and masks
- [`result/`](result): generated figures

## How To Run

From the repository root, run:

```bash
python sliding_demo/vortex_sheet_fenics.py
```

## Output Files

The demo writes its outputs to [`result/`](result/). For a prefix such as `Vortexsheet_fenics_rectangle`, the main files are:

- `*_energies.png`
- `*_result.png`
- `*_deformed_image.png`
- `*_deformed_grid.png`
- `*_quiver.png`
- `*_diff_before.png`
- `*_diff_after.png`

