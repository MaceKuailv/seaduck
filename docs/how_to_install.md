# How to install

The package can be installed simply by

```shell
pip install seaduck
```

or, alternatively

```shell
conda install -c conda-forge seaduck
```

or, alternatively (for latest code development), see [a guide for new contibutors](guide_for_developer.md).

## Optional dependencies

- numba

  - Compile and make everything faster!

- matplotlib

  - Plot the shape of interpolation kernel.

- pandas

  - Show variable aliasing in an intuitive way.

- pooch, zarr, cartopy

  - Needed for running the tutorial examples.

## Required dependencies

- python
- numpy
- xarray
- dask
