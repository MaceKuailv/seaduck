import copy
import logging
import warnings

import numpy as np

from seaduck.eulerian import Position
from seaduck.kernel_weight import KnW
from seaduck.ocedata import RelCoord
from seaduck.runtime_conf import compileable
from seaduck.utils import find_rel, find_rx_ry_oceanparcel, rel2latlon, to_180

DEG2M = 6271e3 * np.pi / 180


@compileable
def increment(t, u, du):
    """Find how far it will travel in duration t.

    For a one dimensional particle with speed u and speed derivative du,
    find how far it will travel in duration t.

    Parameters
    ----------
    t: float, numpy.ndarray
        The time duration
    u: float, numpy.ndarray
        The velocity defined at the starting point.
    du: float, numpy.ndarray
        The velocity gradient. Assumed to be constant.
    """
    incr = u / du * (np.exp(du * t) - 1)
    no_gradient = np.abs(du) < 1e-12
    incr[no_gradient] = (u * t)[no_gradient]
    return incr


def stationary(t, u, du, x0):
    """Find the final position after time t.

    For a one dimensional Particle with speed u and speed derivative du
    starting at x0, find the final position after time t.
    "Stationary" means that we are assuming there is no time dependency.

    Parameters
    ----------
    t: float, numpy.ndarray
        The time duration
    u: float, numpy.ndarray
        The velocity defined at the starting point.
    du: float, numpy.ndarray
        The velocity gradient. Assumed to be constant.
    x0: float, numpy.ndarray
        The starting position.
    """
    incr = increment(t, u, du)
    return incr + x0


@compileable
def stationary_time(u, du, x0):
    """Find the amount of time to leave the cell.

    Find the amount of time it needs for a Particle to hit x = -0.5 and 0.5.
    The time could be negative.

    Parameters
    ----------
    u: numpy.ndarray
        The velocity defined at the starting point.
    du: numpy.ndarray
        The velocity gradient. Assumed to be constant.
    x0: numpy.ndarray
        The starting position.

    Returns
    -------
    tl: numpy.ndarray
        The time it takes to hit -0.5.
    tr: numpy.ndarray
        The time it takes to hit 0.5
    """
    tl = np.log(1 - du / u * (0.5 + x0)) / du
    tr = np.log(1 + du / u * (0.5 - x0)) / du
    no_gradient = np.abs(du) < 1e-12
    tl[no_gradient] = (-x0[no_gradient] - 0.5) / u[no_gradient]
    tr[no_gradient] = (0.5 - x0[no_gradient]) / u[no_gradient]
    return tl, tr


def time2wall(xs, us, dus):
    """Apply stationary_time three times for all three dimensions."""
    ts = []
    for i in range(3):
        tl, tr = stationary_time(us[i], dus[i], xs[i])
        ts.append(tl)
        ts.append(tr)
    return ts


def which_early(tf, ts):
    """Find out which event happens first.

    We are trying to integrate the Particle to time tf.
    The first event is either:
    1. tf is reached before reaching a wall
    2. ts[i] is reached, and a Particle hit a wall. ts[i]*tf>0.

    Parameters
    ----------
    tf: float, numpy.ndarray
        The final time
    ts: list
        The list of events calculated using time2wall
    """
    ts.append(np.ones(len(ts[0])) * tf)  # float or array both ok
    t_directed = np.array(ts) * np.sign(tf)
    t_directed[np.isnan(t_directed)] = np.inf
    t_directed[t_directed <= 0] = np.inf
    tend = t_directed.argmin(axis=0)
    the_t = np.array([ts[te][i] for i, te in enumerate(tend)])
    return tend, the_t


uvkernel = np.array([[0, 0], [1, 0], [0, 1]])
ukernel = np.array([[0, 0], [1, 0]])
vkernel = np.array([[0, 0], [0, 1]])
wkernel = np.array([[0, 0]])
udoll = [[0, 1]]
vdoll = [[0, 2]]
wdoll = [[0]]
ktype = "interp"
h_order = 0
wknw = KnW(kernel=wkernel, inheritance=None, vkernel="linear", ignore_mask=True)
uknw = KnW(kernel=uvkernel, inheritance=udoll, ignore_mask=True)
vknw = KnW(kernel=uvkernel, inheritance=vdoll, ignore_mask=True)
dwknw = KnW(kernel=wkernel, inheritance=None, vkernel="dz", ignore_mask=True)
duknw = KnW(
    kernel=uvkernel, inheritance=udoll, hkernel="dx", h_order=1, ignore_mask=True
)
dvknw = KnW(
    kernel=uvkernel, inheritance=vdoll, hkernel="dy", h_order=1, ignore_mask=True
)

# scalar style kernel for datasets without face
uknw_s = KnW(kernel=ukernel, inheritance=None, ignore_mask=True)
vknw_s = KnW(kernel=vkernel, inheritance=None, ignore_mask=True)
duknw_s = KnW(
    kernel=ukernel, inheritance=None, hkernel="dx", h_order=1, ignore_mask=True
)
dvknw_s = KnW(
    kernel=vkernel, inheritance=None, hkernel="dy", h_order=1, ignore_mask=True
)


class Particle(Position):
    """Lagrangian particle object.

    The Lagrangian particle object. Simply a eulerian Position object
    that know how to move itself.

    Parameters
    ----------
    kwarg: dict
        The keyword argument that feed into Position.from_latlon method
    uname, vname, wname: str
        The variable names for the velocity/mass-transport components.
        If transport is true, pass in names of the volume/mass transport
        across cell wall in m^3/3
        else,  just pass something that is in m/s
    dont_fly: Boolean
        Sometimes there is non-zero vertical velocity at sea surface.
        dont_fly = True set that to zero.
        An error may occur depends on the situation if set otherwise.
    save_raw: Boolean
        Whether to record the analytical history of all particles in an
        unstructured list.
    transport: Boolean
        If transport is true, pass in names of the volume/mass transport
        across cell wall in m^3/3
        else,  just pass velocity that is in m/s
    callback: function that take Particle as input
        A callback function that takes Particle as input. Return boolean
        array that determines which Particle should still be going.
        Users can also define customized functions here.
    max_iteration: int
        The number of analytical steps allowed for the to_next_stop
        method.
    """

    def __init__(
        self,  # 10MB
        uname="UVELMASS",
        vname="VVELMASS",
        wname="WVELMASS",
        dont_fly=True,
        save_raw=False,
        transport=False,
        callback=None,
        max_iteration=200,
        **kwarg,
    ):
        Position.__init__(self)
        self.from_latlon(**kwarg)
        if self.ocedata.readiness["Zl"]:
            self.rel.update(self.ocedata.find_rel_vl_lin(self.dep))
        else:
            (self.izl_lin, self.rzl_lin, self.dzl_lin, self.bzl_lin) = (
                None for i in range(4)
            )
        try:
            self.px, self.py = self.get_px_py()
        except AttributeError:
            pass
        self.uname = uname
        self.vname = vname
        self.wname = wname

        # set up which kernels to use:
        self.wknw = wknw
        self.dwknw = dwknw
        if self.face is not None:
            self.uknw = uknw
            self.duknw = duknw
            self.vknw = vknw
            self.dvknw = dvknw
        else:
            self.uknw = uknw_s
            self.duknw = duknw_s
            self.vknw = vknw_s
            self.dvknw = dvknw_s

        #  user defined function to stop integration.
        self.callback = callback

        # whether u,v,w is in m^3/s or m/s
        self.transport = transport
        if self.transport:
            try:
                self.ocedata["Vol"]
            except KeyError:
                if self.ocedata.readiness["Zl"]:
                    self.ocedata["Vol"] = np.array(
                        self.ocedata._ds["drF"] * self.ocedata._ds["rA"]
                    )
                else:
                    self.ocedata["Vol"] = np.array(self.ocedata._ds["rA"])

        # whether or not setting the w at the surface
        # just to prevent particles taking off
        self.dont_fly = dont_fly
        if dont_fly:
            if wname is not None:
                self.ocedata[wname].loc[{"Zl": 0}] = 0
        self.too_large = self.ocedata.too_large
        self.max_iteration = max_iteration

        if self.too_large:  # pragma: no cover
            pass
        else:
            self.update_uvw_array()
        (self.u, self.v, self.w, self.du, self.dv, self.dw, self.vol) = (
            np.zeros(self.N).astype(float) for i in range(7)
        )
        if self.transport:
            self.get_vol()

        self.get_u_du()
        self.fillna()

        self.save_raw = save_raw
        if self.save_raw:
            self.itlist = [[] for i in range(self.N)]
            self.fclist = [[] for i in range(self.N)]
            self.iylist = [[] for i in range(self.N)]
            self.izlist = [[] for i in range(self.N)]
            self.ixlist = [[] for i in range(self.N)]
            self.rxlist = [[] for i in range(self.N)]
            self.rylist = [[] for i in range(self.N)]
            self.rzlist = [[] for i in range(self.N)]
            self.ttlist = [[] for i in range(self.N)]
            self.uulist = [[] for i in range(self.N)]
            self.vvlist = [[] for i in range(self.N)]
            self.wwlist = [[] for i in range(self.N)]
            self.dulist = [[] for i in range(self.N)]
            self.dvlist = [[] for i in range(self.N)]
            self.dwlist = [[] for i in range(self.N)]
            self.xxlist = [[] for i in range(self.N)]
            self.yylist = [[] for i in range(self.N)]
            self.zzlist = [[] for i in range(self.N)]

    def update_uvw_array(self):
        """Update the prefetched velocity arrays.

        The way to do it is slightly different for dataset with time
        dimensions and those without.
        """
        uname = self.uname
        vname = self.vname
        wname = self.wname
        if "time" not in self.ocedata[uname].dims:
            try:
                assert isinstance(self.uarray, np.ndarray)
                assert isinstance(self.varray, np.ndarray)
                if self.wname is not None:
                    assert isinstance(self.warray, np.ndarray)
            except (AttributeError, AssertionError):
                self.uarray = np.array(self.ocedata[uname])
                self.varray = np.array(self.ocedata[vname])
                if self.wname is not None:
                    self.warray = np.array(self.ocedata[wname])
                    if self.dont_fly:
                        # I think it's fine
                        self.warray[0] = 0.0
                # else:
                #     self.warray = None
        else:
            self.itmin = int(np.min(self.it))
            self.itmax = int(np.max(self.it))
            if self.itmax != self.itmin:
                self.uarray = np.array(self.ocedata[uname][self.itmin : self.itmax + 1])
                self.varray = np.array(self.ocedata[vname][self.itmin : self.itmax + 1])
                if self.wname is not None:
                    self.warray = np.array(
                        self.ocedata[wname][self.itmin : self.itmax + 1]
                    )
                # else:
                #     self.warray = None
            else:
                self.uarray = np.array(self.ocedata[uname][[self.itmin]])
                self.varray = np.array(self.ocedata[vname][[self.itmin]])
                if self.wname is not None:
                    self.warray = np.array(self.ocedata[wname][[self.itmin]])
                # else:
                #     self.warray = None
            if self.dont_fly:
                if self.wname is not None:
                    # I think it's fine
                    self.warray[:, 0] = 0.0

    def get_vol(self):
        """Read in the volume of the cell.

        For particles that has transport = True,
        volume of the cell is needed for the integration.
        This method read the volume that is calculated at __init__.
        """
        index = []
        if self.ocedata.readiness["Zl"]:
            index.append(self.izl_lin - 1)
        if self.face is not None:
            index.append(self.face)
        index += [self.iy, self.ix]
        index = tuple(index)
        self.vol = self.ocedata["Vol"][index]

    def get_u_du(self):
        """Read the velocity at particle position.

        Read the velocity and velocity derivatives in all three dimensions
        using the interpolate method with the default kernel.
        Read eulerian.Position.interpolate for more detail.
        """
        if self.too_large:  # pragma: no cover
            prefetched = None
            i_min = None
        else:
            if "time" not in self.ocedata[self.uname].dims:
                ifirst = 0
            else:
                ifirst = self.itmin
            i_min = [0 for i in self.uarray.shape]
            i_min[0] = ifirst
            i_min = tuple(i_min)

            if self.wname is None:
                self.warray = None
            prefetched = [
                self.warray,
                self.warray,
                (self.uarray, self.varray),
                (self.uarray, self.varray),
            ]

        try:
            self.iz = self.izl_lin - 1
        except (TypeError, AttributeError):
            pass

        [w, dw, (u, v), (du, dv)] = self.interpolate(
            [
                self.wname,
                self.wname,
                (self.uname, self.vname),
                (self.uname, self.vname),
            ],
            [self.wknw, self.dwknw, (self.uknw, self.vknw), (self.duknw, self.dvknw)],
            prefetched=prefetched,
            i_min=i_min,
            vec_transform=False,
        )

        if self.wname is None:
            w = np.zeros_like(u)
            dw = np.zeros_like(u)

        if not self.transport:
            self.u = u / self.dx
            self.v = v / self.dy
            self.du = du / self.dx
            self.dv = dv / self.dy

        else:
            self.u = u / self.vol
            self.v = v / self.vol
            self.du = du / self.vol
            self.dv = dv / self.vol

        if self.wname is not None:
            if not self.transport:
                self.w = w / self.dzl_lin
                self.dw = dw / self.dzl_lin
            else:
                self.w = w / self.vol
                self.dw = dw / self.vol
        self.fillna()

    def fillna(self):
        """Fill the np.nan values to nan.

        This is just to let those in rock stay in rock.
        """
        np.nan_to_num(self.u, copy=False)
        np.nan_to_num(self.v, copy=False)
        np.nan_to_num(self.w, copy=False)
        np.nan_to_num(self.du, copy=False)
        np.nan_to_num(self.dv, copy=False)
        np.nan_to_num(self.dw, copy=False)

    def note_taking(self, which=None):
        """Record raw data into list of lists.

        This method is only called in save_raw = True particles.
        This method will note done the raw info of the particle
        trajectories.
        With those info, one could reconstruct the analytical
        trajectories to arbitrary position.
        """
        if which is None:
            which = np.ones(self.N).astype(bool)
        where = np.where(which)[0]
        try:
            self.ttlist
        except AttributeError as exc:
            raise AttributeError("This is not a particle_rawlist object") from exc
        for i in where:
            if self.face is not None:
                self.fclist[i].append(self.face[i])
            if self.it is not None:
                self.itlist[i].append(self.it[i])
            self.iylist[i].append(self.iy[i])
            if self.izl_lin is not None:
                self.izlist[i].append(self.izl_lin[i])
            self.ixlist[i].append(self.ix[i])
            self.rxlist[i].append(self.rx[i])
            self.rylist[i].append(self.ry[i])
            if self.rzl_lin is not None:
                self.rzlist[i].append(self.rzl_lin[i])
            self.ttlist[i].append(self.t[i])
            self.uulist[i].append(self.u[i])
            self.vvlist[i].append(self.v[i])
            self.wwlist[i].append(self.w[i])
            self.dulist[i].append(self.du[i])
            self.dvlist[i].append(self.dv[i])
            self.dwlist[i].append(self.dw[i])
            self.xxlist[i].append(self.lon[i])
            self.yylist[i].append(self.lat[i])
            self.zzlist[i].append(self.dep[i])

    def empty_lists(self):
        """Empty the lists.

        Some times the raw-data list get too long,
        It would be necessary to dump the data,
        and empty the lists containing the raw data.
        This method does the latter.
        """
        self.itlist = [[] for i in range(self.N)]
        self.fclist = [[] for i in range(self.N)]
        self.iylist = [[] for i in range(self.N)]
        self.izlist = [[] for i in range(self.N)]
        self.ixlist = [[] for i in range(self.N)]
        self.rxlist = [[] for i in range(self.N)]
        self.rylist = [[] for i in range(self.N)]
        self.rzlist = [[] for i in range(self.N)]
        self.ttlist = [[] for i in range(self.N)]
        self.uulist = [[] for i in range(self.N)]
        self.vvlist = [[] for i in range(self.N)]
        self.wwlist = [[] for i in range(self.N)]
        self.dulist = [[] for i in range(self.N)]
        self.dvlist = [[] for i in range(self.N)]
        self.dwlist = [[] for i in range(self.N)]
        self.xxlist = [[] for i in range(self.N)]
        self.yylist = [[] for i in range(self.N)]
        self.zzlist = [[] for i in range(self.N)]

    def _out_of_bound(self):  # pragma: no cover
        """Return particles that are out of the cell bound.

        This is most likely due to numerical error of one sort or another.
        If local cartesian is used, there would be more out_of_bound error.
        """
        x_out = np.logical_or(self.rx > 0.5, self.rx < -0.5)
        y_out = np.logical_or(self.ry > 0.5, self.ry < -0.5)
        if self.rzl_lin is not None:
            z_out = np.logical_or(self.rzl_lin > 1, self.rzl_lin < 0)
        else:
            z_out = False
        return np.logical_or(np.logical_or(x_out, y_out), z_out)

    def trim(self, tol=1e-12):
        """Move the particles from outside the cell into the cell.

        At the same time change the velocity accordingly.
        In the mean time, creating some negiligible error in time.

        Parameters
        ----------
        tol: float
            The relative tolerance when particles is significantly
            close to the cell.
        """
        # tol = 1e-6 # about 10 m horizontal for 1 degree
        if logging.DEBUG >= logging.root.level:  # pragma: no cover
            xmax = np.nanmax(self.rx)
            xmin = np.nanmin(self.rx)
            ymax = np.nanmax(self.ry)
            ymin = np.nanmin(self.ry)

            logging.debug("converting %s to 0.5", xmax)
            logging.debug("converting %s to -0.5", xmin)
            logging.debug("converting %s to 0.5", ymax)
            logging.debug("converting %s to -0.5", ymin)

            if self.rzl_lin is not None:
                zmax = np.nanmax(self.rzl_lin)
                zmin = np.nanmin(self.rzl_lin)
                logging.debug("converting %s to 1", zmax)
                logging.debug("converting %s to 0", zmin)

        # if xmax>=0.5-tol:
        where = self.rx >= 0.5 - tol
        cdx = (0.5 - tol) - self.rx[where]
        self.rx[where] += cdx
        self.u[where] += self.du[where] * cdx
        # if xmin<=-0.5+tol:
        where = self.rx <= -0.5 + tol
        cdx = (-0.5 + tol) - self.rx[where]
        self.rx[where] += cdx
        self.u[where] += self.du[where] * cdx
        # if ymax>=0.5-tol:
        where = self.ry >= 0.5 - tol
        cdx = (0.5 - tol) - self.ry[where]
        self.ry[where] += cdx
        self.v[where] += self.dv[where] * cdx
        # if ymin<=-0.5+tol:
        where = self.ry <= -0.5 + tol
        cdx = (-0.5 + tol) - self.ry[where]
        self.ry[where] += cdx
        self.v[where] += self.dv[where] * cdx
        if self.rzl_lin is not None:
            np.nanmax(self.rzl_lin)
            np.nanmin(self.rzl_lin)
            # if zmax>=1.-tol:
            where = self.rzl_lin >= 1.0 - tol
            cdx = (1.0 - tol) - self.rzl_lin[where]
            self.rzl_lin[where] += cdx
            self.w[where] += self.dw[where] * cdx
            # if zmin<=-0.+tol:
            where = self.rzl_lin <= -0.0 + tol
            cdx = (-0.0 + tol) - self.rzl_lin[where]
            self.rzl_lin[where] += cdx
            self.w[where] += self.dw[where] * cdx

    def _contract(self):  # pragma: no cover
        """Warp time to move particle into cell.

        If particles are not in the cell,
        perform some timewarp to put them as close to the cell as possible.
        This is not used in the main routine. Because it was not deemed
        worthy of the computational cost.
        However, it might be reintroduced in latter versions
        as an option for users that requires more accuracy.
        """
        max_time = 1e3
        out = self._out_of_bound()
        # out = np.logical_and(out,u!=0)
        if self.rzl_lin is not None:
            xs = [self.rx[out], self.ry[out], self.rzl_lin[out] - 1 / 2]
        else:
            x_ = self.rx[out]
            xs = [x_, self.ry[out], np.zeros_like(x_)]
        us = [self.u[out], self.v[out], self.w[out]]
        dus = [self.du[out], self.dv[out], self.dw[out]]
        tmin = -np.ones_like(self.rx[out]) * np.inf
        tmax = np.ones_like(self.rx[out]) * np.inf
        for i in range(3):
            tl, tr = stationary_time(us[i], dus[i], xs[i])
            np.nan_to_num(tl, copy=False)
            np.nan_to_num(tr, copy=False)
            tmin = np.maximum(tmin, np.minimum(tl, tr))
            tmax = np.minimum(tmax, np.maximum(tl, tr))
        #         dead = tmin > tmax

        contract_time = (tmin + tmax) / 2
        contract_time = np.maximum(-max_time, contract_time)
        contract_time = np.maximum(max_time, contract_time)

        np.nan_to_num(contract_time, copy=False, posinf=0, neginf=0)

        con_x = []
        for i in range(3):
            con_x.append(stationary(contract_time, us[i], dus[i], 0))

        cdx = np.nan_to_num(con_x[0])
        cdy = np.nan_to_num(con_x[1])
        cdz = np.nan_to_num(con_x[2])

        self.rx[out] += cdx
        self.ry[out] += cdy
        if self.rzl_lin is not None:
            self.rzl_lin[out] += cdz

        self.u[out] += cdx * self.du[out]
        self.v[out] += cdy * self.dv[out]
        self.w[out] += cdz * self.dw[out]

        self.t[out] += contract_time

    def _extract_velocity_position(self, tf):
        """See analytical_step."""
        if self.rzl_lin is not None:
            xs = [self.rx, self.ry, self.rzl_lin - 1 / 2]
        else:
            x_temp = self.rx
            xs = [x_temp, self.ry, np.zeros_like(x_temp)]
        us = [self.u, self.v, self.w]
        dus = [self.du, self.dv, self.dw]
        return us, dus, xs

    def _move_within_cell(self, the_t, us, dus, xs):
        self.t += the_t
        new_x = []
        new_u = []
        for i in range(3):
            x_move = stationary(the_t, us[i], dus[i], 0)
            new_u.append(us[i] + dus[i] * x_move)
            new_x.append(x_move + xs[i])

        for rr in new_x:
            if np.logical_or(rr > 0.6, rr < -0.6).any():
                where = np.where(np.logical_or(rr > 0.6, rr < -0.6))[0][0]
                raise ValueError(
                    f"Particle way out of bound."
                    # f"tend = {tend[where]},"
                    f" the_t = {the_t[where]},"
                    f" rx = {new_x[0][where]},ry = {new_x[1][where]},rz = {new_x[2][where]}"
                    f"start with u = {self.u[where]}, du = {self.du[where]}, x={self.rx[where]}"
                    f"start with v = {self.v[where]}, dv = {self.dv[where]}, y={self.ry[where]}"
                    f"start with w = {self.w[where]}, dw = {self.dv[where]}, z={self.rzl_lin[where]}"
                )
        return new_x, new_u

    def _sync_latlondep_before_cross(self):
        if self.rzl_lin is not None:
            self.dep = self.bzl_lin + self.dzl_lin * self.rzl_lin
        # Otherwise, keep depth the same.
        try:
            px, py = self.px, self.py
            w = self.get_f_node_weight()
            self.lon = np.einsum("nj,nj->n", w, px.T)
            self.lat = np.einsum("nj,nj->n", w, py.T)
        except AttributeError:
            self.lon, self.lat = rel2latlon(
                self.rx,
                self.ry,
                self.cs,
                self.sn,
                self.dx,
                self.dy,
                self.bx,
                self.by,
            )

    def analytical_step(self, tf):
        """Integrate the particle with velocity.

        The core method.
        A set of particles trying to integrate for time tf
        (could be negative).
        at the end of the call, every particle are either:
        1. ended up somewhere within the cell after time tf.
        2. ended up on a cell wall before
        (if tf is negative, then "after") tf.

        Parameters
        ----------
        tf: float, numpy.ndarray
            The longest duration of the simulation for each particle.
        """
        if isinstance(tf, float):
            tf = np.array([tf for i in range(self.N)])
        us, dus, xs = self._extract_velocity_position(tf)

        ts = time2wall(xs, us, dus)

        tend, the_t = which_early(tf, ts)

        new_x, new_u = self._move_within_cell(the_t, us, dus, xs)

        # Could potentially move this block all the way back
        self.rx, self.ry, temp = new_x
        if self.rzl_lin is not None:
            self.rzl_lin = temp + 1 / 2

        self.u, self.v, self.w = new_u

        self._sync_latlondep_before_cross()

        return tend

    def _cross_cell_wall_index(self, tend):
        type1 = tend <= 3
        translate = {0: 2, 1: 3, 2: 1, 3: 0}
        # left  # right  # down  # up
        trans_tend = np.array([translate[i] for i in tend[type1]])
        if self.face is not None:
            tface, tiy, tix, tiz = (
                self.face.astype(int),
                self.iy.astype(int),
                self.ix.astype(int),
                self.izl_lin.astype(int),
            )
            tface[type1], tiy[type1], tix[type1] = self.tp.ind_tend_vec(
                (tface[type1], tiy[type1], tix[type1]), trans_tend
            )
        else:
            tiy, tix = (
                self.iy.astype(int),
                self.ix.astype(int),
            )
            if self.izl_lin is not None:  # pragema: no cover
                tiz = self.izl_lin.astype(int)
            else:
                tiz = (np.ones_like(tiy) * (-1)).astype(int)
            tiy[type1], tix[type1] = self.tp.ind_tend_vec(
                (tiy[type1], tix[type1]), trans_tend
            )
        type2 = tend == 4
        tiz[type2] += 1
        type3 = tend == 5
        tiz[type3] -= 1

        # investigate stuck
        #         now_masked = maskc[tiz-1,tface,tiy,tix]==0
        #         if now_masked.any():
        #             wrong_ind = (np.where(now_masked))[0]
        #             print(wrong_ind)
        #             print((tiz-1)[wrong_ind],tface[wrong_ind],
        #             tiy[wrong_ind],tix[wrong_ind])
        #             print('rx',[xs[i][wrong_ind] for i in range(3)])
        #             print('u',[us[i][wrong_ind] for i in range(3)])
        #             print('du',[dus[i][wrong_ind] for i in range(3)])
        #             print(tend[wrong_ind])
        #             print(t_directed[:,wrong_ind])
        #             print('stuck!')
        #             raise ValueError('ahhhhh!')
        if self.face is not None:
            self.face, self.iy, self.ix = (tface, tiy, tix)
        else:
            self.iy, self.ix = tiy, tix
        if self.izl_lin is not None:
            self.izl_lin = tiz

    def _cross_cell_wall_read(self):
        if self.face is not None:
            horizontal_index = (self.face, self.iy, self.ix)
        else:
            horizontal_index = (self.iy, self.ix)

        if self.ocedata.readiness["h"] == "local_cartesian":
            self.bx, self.by = (
                self.ocedata.XC[horizontal_index],
                self.ocedata.YC[horizontal_index],
            )
            self.cs, self.sn = (
                self.ocedata.CS[horizontal_index],
                self.ocedata.SN[horizontal_index],
            )
            self.dx, self.dy = (
                self.ocedata.dX[horizontal_index],
                self.ocedata.dY[horizontal_index],
            )
        elif self.ocedata.readiness["h"] == "oceanparcel":
            self.bx, self.by = (
                self.ocedata.XC[horizontal_index],
                self.ocedata.YC[horizontal_index],
            )
            self.px, self.py = self.get_px_py()

        elif self.ocedata.readiness["h"] == "rectilinear":
            self.bx = self.ocedata.lon[self.ix]
            self.by = self.ocedata.lat[self.iy]
            self.cs = np.ones_like(self.bx)
            self.sn = np.zeros_like(self.bx)
            self.dx = self.ocedata.dlon[self.ix] * np.cos(self.by * np.pi / 180)
            self.dy = self.ocedata.dlat[self.iy]

        if self.izl_lin is not None:
            self.bzl_lin = self.ocedata.Zl[self.izl_lin]
            self.dzl_lin = self.ocedata.dZl[self.izl_lin - 1]
        if self.dz is not None:
            self.dz = self.ocedata.dZ[self.iz]

    def _cross_cell_wall_rel(self):
        if self.ocedata.readiness["Z"]:
            self.rel.update(self.ocedata.find_rel_v(self.dep))
        if self.rzl_lin is not None:
            self.rzl_lin = (self.dep - self.bzl_lin) / self.dzl_lin

        if self.ocedata.readiness["h"] == "oceanparcel":
            self.rx, self.ry = find_rx_ry_oceanparcel(
                self.lon, self.lat, self.px, self.py
            )
        else:
            #         if True:
            dlon = to_180(self.lon - self.bx)
            dlat = to_180(self.lat - self.by)
            self.rx = (
                (dlon * np.cos(self.by * np.pi / 180) * self.cs + dlat * self.sn)
                * DEG2M
                / self.dx
            )
            self.ry = (
                (dlat * self.cs - dlon * self.sn * np.cos(self.by * np.pi / 180))
                * DEG2M
                / self.dy
            )

    def cross_cell_wall(self, tend):
        """Update properties after particles cross wall.

        A wall event is triggered when particle reached the wall.
        This method handle the coords translation as a particle cross
        a wall.
        """
        self._cross_cell_wall_index(tend)
        self._cross_cell_wall_read()
        self._cross_cell_wall_rel()

    def deepcopy(self):
        """Return a clone of the object.

        The object is a Position object, and thus cannot move any more.
        """
        p = Position()
        p.ocedata = self.ocedata
        p.N = self.N
        varsdict = vars(self)
        keys = varsdict.keys()
        for i in keys:
            item = varsdict[i]
            if isinstance(item, np.ndarray):
                if len(item.shape) == 1:
                    setattr(p, i, copy.deepcopy(item))
            elif isinstance(item, (list, RelCoord)):
                setattr(p, i, copy.deepcopy(item))
        return p

    def to_next_stop(self, t1):
        """Integrate all particles towards time tl.

        This is done by repeatedly calling analytical step.
        Or at least try to do so before maximum_iteration is reached.
        If the maximum time is reached,
        we also force all particle's internal clock to be tl.

        Parameters
        ----------
        t1: float
            The final time relative to 1970-01-01 in seconds.
        """
        tol = 1e-4
        tf = t1 - self.t
        bool_todo = abs(tf) > tol
        if self.callback is not None:
            bool_todo = np.logical_and(bool_todo, self.callback(self))
        int_todo = np.where(bool_todo)[0]
        if len(int_todo) == 0:
            logging.info("Nothing left to simulate")
            return
        tf_used = tf[int_todo]
        trim_tol = 1e-12
        for i in range(self.max_iteration):
            if i > 50:
                trim_tol = 1e-3
            elif i > 30:
                trim_tol = 1e-6
            elif i > 10:
                trim_tol = 1e-10
            logging.debug(len(int_todo), "left")
            sub = self.subset(int_todo)
            sub.trim(tol=trim_tol)
            tend = sub.analytical_step(tf_used)
            # TODO: fix save-raw here.
            if self.save_raw:
                # record the moment just before crossing the wall
                # or the moment reaching destination.
                self.note_taking(int_todo)
            sub.cross_cell_wall(tend)

            if self.transport:
                sub.get_vol()
            sub.get_u_du()
            self.update_from_subset(sub, int_todo)
            tf_used = t1 - sub.t
            bool_todo = abs(tf_used) > tol
            if self.callback is not None:
                bool_todo = np.logical_and(bool_todo, self.callback(sub))
            int_todo = int_todo[bool_todo]
            tf_used = tf_used[bool_todo]
            if len(int_todo) == 0:
                break
            if self.save_raw:
                # record those who cross the wall
                self.note_taking(int_todo)

        if i == self.max_iteration - 1:  # pragma: no cover
            warnings.warn("maximum iteration count reached")
        self.t = np.ones(self.N) * t1
        if self.ocedata.readiness["time"]:
            before_first = self.t < self.ocedata.time_midp[0]
            self.it[before_first] = 0
            self.it[~before_first], _, _, _ = find_rel(
                self.t[~before_first], self.ocedata.time_midp
            )
            self.it[~before_first] += 1

    def to_list_of_time(
        self, normal_stops, update_stops="default", return_in_between=True
    ):
        """Integrate the particles to a list of time.

        Parameters
        ----------
        normal_stops: iterable
            The time steps that user request a output
        update_stops: iterable, or 'default'
            The time steps that uvw array changes in the model.
            If 'default' is set,
            the method is going to figure it out automatically.
        return_in_between: Boolean
            Users can get the values of update_stops free of computational
            cost.We understand that user may sometimes don't want those in
            the output.In that case, it that case, set it to be False,
            and the output will all be at normal_stops.

        Returns
        -------
        stops: list
            The list of stops.
            It is the combination of normal_stops and output_stops by
            default. f return_in_between is set to be False,
            this is then the same as normal stops.
        to_return: list
            A list deep copy of particle that inherited
            the interpolate method
            as well as velocity and coords info.
        """
        t_min = np.minimum(np.min(normal_stops), self.t[0])
        t_max = np.maximum(np.max(normal_stops), self.t[0])

        if "time" not in self.ocedata[self.uname].dims:
            pass
        else:
            data_tmin = self.ocedata.ts.min()
            data_tmax = self.ocedata.ts.max()
            if t_min < data_tmin or t_max > data_tmax:
                raise ValueError(
                    "time range not within bound" + f"({data_tmin},{data_tmax})"
                )
        if update_stops == "default":
            try:
                update_stops = self.ocedata.time_midp[
                    np.logical_and(
                        t_min < self.ocedata.time_midp, self.ocedata.time_midp < t_max
                    )
                ]
            except AttributeError as exc:
                raise AttributeError(
                    "time_midp is required for "
                    "update_stops = default,"
                    " but it is not in the dataset, "
                    "either create it or "
                    "specify the update stops."
                ) from exc
        temp = list(zip(normal_stops, np.zeros_like(normal_stops))) + list(
            zip(update_stops, np.ones_like(update_stops))
        )
        temp.sort(key=lambda x: abs(x[0] - self.t[0]))
        stops, update = list(zip(*temp))
        #         return stops,update
        self.get_u_du()
        to_return = []
        for i, tl in enumerate(stops):
            logging.info(np.datetime64(round(tl), "s"))
            if self.save_raw:
                # save the very start of everything.
                self.note_taking()
            self.to_next_stop(tl)
            if update[i]:
                if self.too_large:  # pragma: no cover
                    self.get_u_du()
                elif "time" not in self.ocedata[self.uname].dims:  # pragema: no cover
                    pass
                else:
                    self.update_uvw_array()
                    self.get_u_du()
                if return_in_between:
                    to_return.append(self.deepcopy())
            else:
                to_return.append(self.deepcopy())
            if self.save_raw:
                self.empty_lists()
        return stops, to_return
