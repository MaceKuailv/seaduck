# seaduck.OceInterp

## OceInterp

```python
OceInterp(od, var_list, x, y, z, t, kernel_list=None, lagrangian=False, lagrange_kwarg={}, update_stops='default', return_in_between=True, return_pt_time=True, kernel_kwarg={})
```

Interp for people who just want to take a quick look.

**This is the centerpiece function of the package, through which
you can access almost all of its functionality.**.

**Parameters**

- **od** (*OceInterp.OceData object or xarray.Dataset (limited support for netCDF Dataset)*) -- The dataset to work on.
- **var_list** (*str or list*) -- A list of variable or pair of variables.
- **kernel_list** (*OceInterp.KnW or list of OceInterp.KnW objects, optional*) -- Indicates which kernel to use for each interpolation.
- **x, y, z** (*numpy.ndarray, float*) -- The location of the particles, where x and y are in degrees, and z is in meters (deeper locations are represented by more negative values).
- **t** (*numpy.ndarray, float, string/numpy.datetime64*) -- In the Eulerian scheme, this represents the time of interpolation. In the Lagrangian scheme, it represents the time needed for output.
- **lagrangian** (*bool, default False*) -- Specifies whether the interpolation is done in the Eulerian or Lagrangian scheme.
- **lagrange_kwarg** (*dict, optional*) -- Keyword arguments passed into the OceInterp.lagrangian.Particle object.
- **update_stops** (*None, 'default', or iterable of float*) -- Specifies the time to update the prefetch velocity.
- **return_in_between** (*bool, default True*) -- In Lagrangian mode, this returns the interpolation not only at time t, but also at every point in time when the speed is updated.
- **return_pt_time** (*bool, default True*) -- Specifies whether to return the time of all the steps.
- **kernel_kwarg** (*dict, optional*) -- keyword arguments to pass into seaduck.KnW object.
