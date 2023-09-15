import warnings

import numpy as np

from seaduck.eulerian import Position
from seaduck.kernel_weight import KnW
from seaduck.lagrangian import Particle, uknw, vknw
from seaduck.ocedata import OceData
from seaduck.utils import convert_time

lagrange_token = "__particle."


def OceInterp(
    od,
    var_list,
    x,
    y,
    z,
    t,
    kernel_list=None,
    lagrangian=False,
    lagrange_kwarg={},
    update_stops="default",
    return_in_between=True,
    return_pt_time=True,
    kernel_kwarg={},
):
    """Interp for people who just want to take a quick look.

    **This is the centerpiece function of the package, through which
    you can access almost all of its functionality.**.

    Parameters
    ----------
    od: OceInterp.OceData object or xarray.Dataset (limited support for netCDF Dataset)
        The dataset to work on.
    var_list: str or list
        A list of variable or pair of variables.
    kernel_list: OceInterp.KnW or list of OceInterp.KnW objects, optional
        Indicates which kernel to use for each interpolation.
    x, y, z: numpy.ndarray, float
        The location of the particles, where x and y are in degrees,
        and z is in meters (deeper locations are represented by more negative values).
    t: numpy.ndarray, float, string/numpy.datetime64
        In the Eulerian scheme, this represents the time of interpolation.
        In the Lagrangian scheme, it represents the time needed for output.
    lagrangian: bool, default False
        Specifies whether the interpolation is done in the Eulerian or Lagrangian scheme.
    lagrange_kwarg: dict, optional
        Keyword arguments passed into the OceInterp.lagrangian.Particle object.
    update_stops: None, 'default', or iterable of float
        Specifies the time to update the prefetch velocity.
    return_in_between: bool, default True
        In Lagrangian mode, this returns the interpolation not only at time t,
        but also at every point in time when the speed is updated.
    return_pt_time: bool, default True
        Specifies whether to return the time of all the steps.
    kernel_kwarg: dict, optional
        keyword arguments to pass into seaduck.KnW object.
    """
    if not isinstance(od, OceData):
        od = OceData(od)

    if isinstance(var_list, dict):
        kernel_list = list(var_list.values())
        var_list = list(var_list.keys())
    elif isinstance(var_list, str):
        var_list = [var_list]
    elif isinstance(var_list, tuple):
        var_list = [var_list]
    elif isinstance(var_list, list):
        pass
    else:
        raise ValueError("var_list type not recognized.")
    if isinstance(kernel_list, list):
        pass
    elif kernel_list is None:
        kernel_list = []
        the_kernel = KnW(**kernel_kwarg)
        for i in var_list:
            if isinstance(i, str):
                kernel_list.append(the_kernel)
            elif isinstance(i, tuple):
                if kernel_kwarg != {}:
                    kernel_list.append((the_kernel, the_kernel))
                else:
                    kernel_list.append((uknw, vknw))
            else:
                raise ValueError(
                    "members of var_list need to be made up of string or tuples"
                )
    if isinstance(t, np.ndarray):
        if np.issubdtype(t.dtype, np.datetime64) or np.issubdtype(t.dtype, str):
            t = np.array([convert_time(some_t) for some_t in t])
    elif isinstance(t, (np.datetime64, str)):
        t = convert_time(t)
    elif isinstance(t, float):
        pass
    else:
        raise ValueError("time format not supported")
    if not lagrangian:
        pt = Position()
        pt.from_latlon(x=x, y=y, z=z, t=t, data=od)
        for i, var in enumerate(var_list):
            if lagrange_token in var:
                raise AttributeError(
                    "__particle variables is only available for Lagrangian Particles"
                )
        to_return = pt.interpolate(var_list, kernel_list)
        return to_return

    else:
        try:
            assert len(t) > 1
        except AssertionError as exc:
            raise ValueError(
                "There needs to be at least two time steps to run the lagrangian Particle"
            ) from exc
        t_start = t[0]
        t_nec = t[1:]
        pt = Particle(
            x=x, y=y, z=z, t=np.ones_like(x) * t_start, data=od, **lagrange_kwarg
        )
        stops, raw = pt.to_list_of_time(
            t_nec, update_stops=update_stops, return_in_between=return_in_between
        )
        to_return = []
        for i, var in enumerate(var_list):
            if var == lagrange_token + "raw":
                to_return.append(raw)
            elif lagrange_token in var:
                sublist = []
                for snap in raw:
                    sublist.append(getattr(snap, var[len(lagrange_token) :]))
                to_return.append(sublist)
            else:
                sublist = []
                for snap in raw:
                    sublist.append(snap.interpolate(var, kernel_list[i]))
                to_return.append(sublist)

        if return_pt_time:
            return stops, to_return
        else:
            if return_in_between:
                warnings.warn("Some of the returns is not on the times you specified.")
            return to_return
