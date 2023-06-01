import logging

import numpy as np

try:  # pragma: no cover
    import pandas as _pd
except ImportError:  # pragma: no cover
    pass
import xarray as xr

from seaduck.topology import Topology
from seaduck.utils import (
    _general_len,
    create_tree,
    find_cs_sn,
    find_rel_h_naive,
    find_rel_h_oceanparcel,
    find_rel_h_rectilinear,
    find_rel_nearest,
    find_rel_time,
    find_rel_z,
)

NO_ALIAS = {
    "dXC": "dxC",
    "dYC": "dyC",
    "dZ": "drC",
    "dXG": "dxG",
    "dYG": "dyG",
    "dZl": "drF",
}


class RelCoord(dict):
    """
    NamedTuple that also has update method.

    This class is used to store the relative coordinates.
    Attributes starts with "i" are indexes of the nearest grid point.
    Attributes starts with "b" are value (time/dep/lat/lon) of the nearest grid point.
    Attributes starts with "d" are distance between the nearest grid point and its
    neighboring point in meters or seconds.
    Attributes starts with "r" are the distance from the point of interest to the nearest
    non-dimensionalized by the "d" variable.
    "cs", "sn" are the cosine and sine of the grid orientation relative to meridian.
    "face" is the face/tile the point is on, if the dataset has such a dimension.

    All of those attributes should be None or 1D numpy array.

    Methods
    -------
    update(other)
        Inheritated from dictionary.
    """

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError as exc:
            raise AttributeError(
                f"'RelCoord' object has no attribute '{attr}'"
            ) from exc

    def __setattr__(self, attr, value):
        self[attr] = value

    def subset(self, which):
        new = RelCoord()
        for var in self.keys():
            if self[var] is not None:
                new[var] = self[var][which]
            else:
                new[var] = None
        return new

    @classmethod
    def create_class(cls, class_name, fields):
        """Create a subclass with predetermined keys."""

        class NewClass(cls):
            __slots__ = ()
            _fields = fields

            def __init__(self, *args, **kwargs):
                if len(args) > len(fields):
                    raise TypeError(
                        f"{class_name} takes {len(fields)} positional arguments"
                        f" but {len(args)} were given"
                    )
                for name, value in zip(fields, args):
                    setattr(self, name, value)
                for name, value in kwargs.items():
                    setattr(self, name, value)

            @classmethod
            def _make(cls, iterable):
                """Make a instance of this class similar to that of collections.namedtuple."""
                return cls(*iterable)

            def __repr__(self):
                field_values = ", ".join(
                    f"{name}={getattr(self, name)!r}" for name in self._fields
                )
                return f"{class_name}({field_values})"

        NewClass.__name__ = class_name
        return NewClass


HRel = RelCoord.create_class(
    "HRel", ["face", "iy", "ix", "rx", "ry", "cs", "sn", "dx", "dy", "bx", "by"]
)
HRel.__doc__ = "Wrap around the horizontal rel-coords. See also RelCoord."
VRel = RelCoord.create_class("VRel", ["iz", "rz", "dz", "bz"])
VRel.__doc__ = (
    "Wrap around the vertical centered nearest rel-coords. See also RelCoord."
)
VLinRel = RelCoord.create_class("VLRel", ["iz_lin", "rz_lin", "dz_lin", "bz_lin"])
VLinRel.__doc__ = (
    "Wrap around the vertical centered linear rel-coords. See also RelCoord."
)
VlRel = RelCoord.create_class("VlRel", ["izl", "rzl", "dzl", "bzl"])
VlRel.__doc__ = (
    "Wrap around the vertical staggered nearest rel-coords. See also RelCoord."
)
VlLinRel = RelCoord.create_class(
    "VlLinRel", ["izl_lin", "rzl_lin", "dzl_lin", "bzl_lin"]
)
VlLinRel.__doc__ = (
    "Wrap around the vertical staggered linear rel-coords. See also RelCoord."
)
TRel = RelCoord.create_class("TRel", ["it", "rt", "dt", "bt"])
TRel.__doc__ = "Wrap around the temporal nearest rel-coords."
TLinRel = RelCoord.create_class("TLinRel", ["it_lin", "rt_lin", "dt_lin", "bt_lin"])
TRel.__doc__ = "Wrap around the temporal linear rel-coords. See also RelCoord."


class OceData:
    """Ocean dataset.

    Class that enables variable aliasing, topology extraction, and 4-D translation
    between latitude-longitude-depth-time grid and relative description.

    Parameters
    ----------
    data: xarray.Dataset
        The dataset to extract grid information, create cKD tree, and Topology object on.
    alias: dict, None, or 'auto'
        1. dict: Map the variable used by this package (key) to
           that used by the dataset (value).
        2. None (default): Do not apply alias.
        3. 'auto' (Not implemented): Automatically generate a list for the alias.
    """

    def __init__(self, data, alias=None, memory_limit=1e7):
        self._ds = data
        self.tp = Topology(data)
        if alias is None:
            self.alias = NO_ALIAS
        elif alias == "auto":
            raise NotImplementedError("auto alias not yet implemented")
        elif isinstance(alias, dict):
            self.alias = alias

        try:
            self.too_large = self._ds["XC"].nbytes > memory_limit
        except KeyError:
            self.too_large = False
        readiness, missing = self.check_readiness()
        self.readiness = readiness
        if readiness:
            self.grid2array()
        else:
            raise ValueError(
                f"""
            use add_missing_variables or set_alias to create {missing},
            then call OceData.grid2array.
            """
            )

    def __setitem__(self, key, item):
        if isinstance(item, xr.DataArray):
            if key in self.alias.keys():
                self._ds[self.alias[key]] = item
            else:
                self._ds[key] = item
        else:
            object.__setattr__(self, key, item)

    def __getitem__(self, key):
        varsdict = vars(self)
        if key in varsdict.keys():
            return object.__getattribute__(self, key)
        else:
            if key in self.alias.keys():
                return self._ds[self.alias[key]]
            else:
                return self._ds[key]

    def check_readiness(self):
        """Return readiness.

        Function that checks what kind of interpolation is supported.

        Returns
        -------
        readiness: dict
            'h': The scheme of horizontal interpolation to be used,
                 including 'oceanparcel', 'local_cartesian', and 'rectilinear'.
            'Z': Whether the dataset has a vertical dimension at the center points.
            'Zl': Whether the dataset has a vertical dimension at staggered points
                 (vertical velocity).
            'time': Whether the dataset has a temporal dimension.
        """
        # TODO: make the check more detailed
        varnames = list(self._ds.data_vars) + list(self._ds.coords)
        readiness = {}
        missing = []
        if all(i in varnames for i in ["XC", "YC", "XG", "YG"]):
            readiness["h"] = "oceanparcel"
            # could be a curvilinear grid that can use oceanparcel style
        elif all(i in varnames for i in ["XC", "YC", "dxG", "dyG", "CS", "SN"]):
            readiness["h"] = "local_cartesian"
            # could be a curvilinear grid that can use local cartesian style
        elif all(i in varnames for i in ["lon", "lat"]):
            ratio = 6371e3 * np.pi / 180
            self.dlon = np.gradient(self["lon"]) * ratio
            self.dlat = np.gradient(self["lat"]) * ratio
            readiness["h"] = "rectilinear"
            # corresponding to a rectilinear dataset
        else:
            readiness["h"] = False
            # readiness['overall'] = False
            missing.append(
                """
            For the most basic use case,
            {XC,YC,XG,YG} or {XC,YC,dxG,dyG,CS,SN} for curvilinear grid;
            or {lon, lat} for rectilinear grid.
            """
            )
            return False, missing
        for _ in ["time", "Z", "Zl"]:
            readiness[_] = (_ in varnames) and (_general_len(self[_]) > 1)

        return readiness, missing

    def _add_missing_grid(self):
        # TODO:
        pass

    def show_alias(self):
        """Print out the alias in a nice pd.DataFrame format."""
        try:
            return _pd.DataFrame.from_dict(
                self.alias, orient="index", columns=["original name"]
            )
        except NameError as exc:
            raise NameError("pandas is needed to perform this function.") from exc

    def _add_missing_cs_sn(self):
        try:
            assert (self["SN"] is not None) and (self["CS"] is not None)
        except (AttributeError, AssertionError):
            xc = np.deg2rad(np.array(self["XC"]))
            yc = np.deg2rad(np.array(self["YC"]))
            cs = np.zeros_like(xc)
            sn = np.zeros_like(xc)
            cs[0], sn[0] = find_cs_sn(yc[0], xc[0], yc[1], xc[1])
            cs[-1], sn[-1] = find_cs_sn(yc[-2], xc[-2], yc[-1], xc[-1])
            cs[1:-1], sn[1:-1] = find_cs_sn(yc[:-2], xc[:-2], yc[2:], xc[2:])
            # it makes no sense to turn it into DataArray again when you already have in memory
            # and you know where this data is defined.
            self["CS"] = cs
            self["SN"] = sn

    def hgrid2array(self):
        """Extract the horizontal grid data into numpy arrays.

        This is done based on the readiness['h'] of the OceData object.
        """
        way = self.readiness["h"]
        if self.too_large:  # pragma: no cover
            logging.warning(
                "Loading grid into memory, it's a large dataset please be patient"
            )

        if way == "oceanparcel":
            for var in ["XC", "YC", "XG", "YG"]:
                self[var] = np.array(self[var]).astype("float32")
            for var in ["rA", "CS", "SN"]:
                try:
                    self[var] = np.array(self[var]).astype("float32")
                except KeyError:
                    logging.info("no %s in dataset, skip", var)
                    self[var] = None
            try:
                self.dX = np.array(self["dXG"]).astype("float32")
                self.dY = np.array(self["dYG"]).astype("float32")
            except KeyError:
                self.dX = None
                self.dY = None
            if self.too_large:  # pragma: no cover
                logging.info("numpy arrays of grid loaded into memory")
            self.tree = create_tree(self.XC, self.YC)
            if self.too_large:  # pragma: no cover
                logging.info("cKD created")

        if way == "local_cartesian":
            for var in ["XC", "YC", "CS", "SN"]:
                self[var] = np.array(self[var]).astype("float32")
            self.dX = np.array(self["dXG"]).astype("float32")
            self.dY = np.array(self["dYG"]).astype("float32")

            if not self.too_large:  # pragma: no cover
                for var in ["XG", "YG", "dXC", "dYC", "rA"]:
                    try:
                        self[var] = np.array(self[var]).astype("float32")
                    except KeyError:
                        logging.info("no %s in dataset, skip", var)
                        self[var] = None
            if self.too_large:  # pragma: no cover
                logging.info("numpy arrays of grid loaded into memory")
            self.tree = create_tree(self.XC, self.YC)
            if self.too_large:  # pragma: no cover
                logging.info("cKD created")

        if way == "rectilinear":
            self.lon = np.array(self["lon"]).astype("float32")
            self.lat = np.array(self["lat"]).astype("float32")

    def vgrid2array(self):
        """Extract the vertical center point grid data into numpy arrays."""
        self.Z = np.array(self["Z"]).astype("float32")
        self.dZ = np.array(self["dZ"]).astype("float32")

    def vlgrid2array(self):
        """Extract the vertical staggered point grid data into numpy arrays."""
        self.Zl = np.array(self["Zl"]).astype("float32")
        self.dZl = np.array(self["dZl"]).astype("float32")

        # special treatment for dZl
        # self.dZl = np.roll(self.dZl,1)
        # self.dZl[0] = 1e-10

    def tgrid2array(self):
        """Extract the temporal grid data into numpy arrays."""
        self.t_base = 0
        self.ts = np.array(self["time"])
        self.ts = (self.ts).astype(float) / 1e9
        try:
            self.time_midp = np.array(self["time_midp"])
            self.time_midp = (self.time_midp).astype(float) / 1e9
        except KeyError:
            self.time_midp = (self.ts[1:] + self.ts[:-1]) / 2

    def grid2array(self):
        """Assemble all the extraction methods."""
        if self.readiness["h"]:
            self.hgrid2array()
        if self.readiness["Z"]:
            self.vgrid2array()
        if self.readiness["Zl"]:
            self.vlgrid2array()
        if self.readiness["time"]:
            self.tgrid2array()

    def find_rel_h(self, x, y):
        """Find the horizontal rel-coordinate.

        Find the horizontal rel-coordinate of the given 4-D position based on readiness['h'].

        Parameters
        ----------
        x, y: np.ndarray
            1D array of longitude and latitude.

        Returns
        -------
        hrel: seaduck.ocedata.HRel object
            A dictionary that defines the horizontal rel-coords
        """
        if self.readiness["h"] == "oceanparcel":
            h_rel_tuple = find_rel_h_oceanparcel(
                x,
                y,
                self.XC,
                self.YC,
                self.dX,
                self.dY,
                self.CS,
                self.SN,
                self.XG,
                self.YG,
                self.tree,
                self.tp,
            )
        elif self.readiness["h"] == "local_cartesian":
            h_rel_tuple = find_rel_h_naive(
                x, y, self.XC, self.YC, self.dX, self.dY, self.CS, self.SN, self.tree
            )
        elif self.readiness["h"] == "rectilinear":
            h_rel_tuple = find_rel_h_rectilinear(x, y, self.lon, self.lat)
        return HRel._make(h_rel_tuple)

    def find_rel_v(self, z):
        """Find the rel-coord based on vertical center grid using the nearest neighbor scheme."""
        iz, rz, dz, bz = find_rel_nearest(z, self.Z)
        return VRel(iz.astype(int), rz, dz, bz)

    def find_rel_v_lin(self, z):
        """Find the rel-coord based on vertical center grid using the 2-point linear scheme."""
        iz, rz, dz, bz = find_rel_z(z, self.Z, self.dZ)
        return VLinRel(iz.astype(int), rz, dz, bz)

    def find_rel_vl(self, z):
        """Find the rel-coord based on vertical staggered grid using the nearest neighbor scheme."""
        iz, rz, dz, bz = find_rel_nearest(z, self.Zl)
        return VlRel(iz.astype(int), rz, dz, bz)

    def find_rel_vl_lin(self, z):
        """Find the rel-coord based on vertical staggered grid using the 2-point linear scheme."""
        iz, rz, dz, bz = find_rel_z(z, self.Zl, self.dZl, dz_above_z=False)
        return VlLinRel(iz.astype(int), rz, dz, bz)

    def find_rel_t(self, t):
        """Find the rel-coord based on the temporal direction using the nearest neighbor scheme."""
        it, rt, dt, bt = find_rel_nearest(t, self.ts)
        return TRel(it.astype(int), rt, dt, bt)

    def find_rel_t_lin(self, t):
        """Find the rel-coord based on the temporal direction using the 2-point linear scheme."""
        it, rt, dt, bt = find_rel_time(t, self.ts)
        return TLinRel(it.astype(int), rt, dt, bt)
