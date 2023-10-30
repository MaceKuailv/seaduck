import copy
import logging

import numpy as np

from seaduck.get_masks import get_masked
from seaduck.kernel_weight import KnW, _find_pk_4d, _translate_to_tendency
from seaduck.ocedata import HRel, OceData, RelCoord, TRel, VlRel, VRel
from seaduck.smart_read import smart_read
from seaduck.utils import (
    _general_len,
    find_px_py,
    get_key_by_value,
    local_to_latlon,
    to_180,
    weight_f_node,
)


def _ind_broadcast(x, ind):
    """Perform a "cartesian product" on two fattened dimensions.

    Parameters
    ----------
    x: numpy.ndarray
        A new dimension
    ind: tuple
        Existing indexes
    """
    n = x.shape[0]
    if len(x.shape) == 1:
        x = x.reshape((n, 1))
    xsp = x.shape
    ysp = ind[0].shape
    final_shape = [n] + list(ysp[1:]) + list(xsp[1:])

    to_return = [np.zeros(final_shape, int) for i in range(len(ind) + 1)]

    dims = len(final_shape)
    ydim = len(ysp) - 1
    trsp = list(range(1, 1 + ydim)) + [0] + list(range(1 + ydim, dims))
    inv = np.argsort(trsp)
    to_return[0] = to_return[0].transpose(trsp)
    to_return[0][:] = x
    to_return[0] = to_return[0].transpose(inv)

    for i in range(1, len(ind) + 1):
        to_return[i] = to_return[i].T
        to_return[i][:] = ind[i - 1].T
        to_return[i] = to_return[i].T
    return to_return


def _partial_flatten(ind):
    """Convert a high dimensional index set to a 2D one."""
    if isinstance(ind, tuple):
        shape = ind[0].shape

        # num_neighbor = 1
        # for i in range(1,len(shape)):
        #     num_neighbor*=shape[i]
        to_return = []
        for i in range(len(ind)):
            to_return.append(ind[i].reshape(shape[0], -1))
        return tuple(to_return)
    elif isinstance(ind, np.ndarray):
        shape = ind.shape
        num_neighbor = 1
        for i in range(1, len(shape)):
            num_neighbor *= shape[i]
        return ind.reshape(shape[0], num_neighbor)
    else:
        raise NotImplementedError("ind type not supported")


def _in_required(name, required):
    """See if a name is in required."""
    if required == "all":
        return True
    else:
        return name in required


def _ind_for_mask(ind, dims):
    """Find the index for masking.

    If dims does not include a vertical dimension, assume to be 0.
    If dims has a temporal dimension, take it away.
    Return the index for masking.
    """
    ind_for_mask = [ind[i] for i in range(len(ind)) if dims[i] not in ["time"]]
    if "Z" not in dims and "Zl" not in dims:
        ind_for_mask.insert(0, np.zeros_like(ind[0]))
    return tuple(ind_for_mask)


def _subtract_prefetch_prefix(ind, prefetch_prefix):
    """Subtract the index prefix from the actual index.

    This is used when one is reading from a prefetched subset of the data.
    """
    temp_ind = []
    for i in range(len(prefetch_prefix)):
        temp_ind.append(ind[i] - prefetch_prefix[i])
    return tuple(temp_ind)


class Position:
    """The Position object that performs the interpolation.

    Create a empty one by default.
    To actually do interpolation, use from_latlon method to tell the ducks where they are.
    """

    def __new__(cls, *arg, **kwarg):
        new_position = object.__new__(cls)
        new_position.rel = RelCoord()
        return new_position

    def __getattr__(self, attr):
        if attr == "rel":
            object.__getattribute__(self, attr)
        elif attr in self.rel.keys():
            return getattr(self.rel, attr)
        else:
            object.__getattribute__(self, attr)

    def __setattr__(self, attr, value):
        if attr == "rel":
            object.__setattr__(self, attr, value)
        elif attr in self.rel.keys():
            setattr(self.rel, attr, value)
        else:
            object.__setattr__(self, attr, value)

    def from_latlon(self, x=None, y=None, z=None, t=None, data=None):
        """Fill in the coord info using lat-lon-dep-time dims.

        Use the methods from the ocedata to transform
        from lat-lon-dep-time coords to rel-coords
        store the output in the Position object.

        Parameters
        ----------
        x,y,z,t: numpy.ndarray, float or None, default None
            1D array of the lat-lon-dep-time coords
        data: OceData object
            The field where the Positions are defined on.
        """
        if data is None:
            try:
                self.ocedata
            except AttributeError as exc:
                raise ValueError("data not provided") from exc
        else:
            if isinstance(data, OceData):
                self.ocedata = data
            else:
                raise ValueError("Input data must be OceData")
        self.tp = self.ocedata.tp
        length = [_general_len(i) for i in [x, y, z, t]]
        self.N = max(length)
        if any(i != self.N for i in length if i > 1):
            raise ValueError("Shapes of input coordinates are not compatible")

        if isinstance(x, (int, float, np.floating)):
            x = np.ones(self.N, float) * x
        if isinstance(y, (int, float, np.floating)):
            y = np.ones(self.N, float) * y
        if isinstance(z, (int, float, np.floating)):
            z = np.ones(self.N, float) * z
        if isinstance(t, (int, float, np.floating)):
            t = np.ones(self.N, float) * t

        for thing in [x, y, z, t]:
            if thing is None:
                continue
            if len(thing.shape) > 1:
                raise ValueError("Input need to be 1D numpy arrays")
        if (x is not None) and (y is not None):
            self.lon = copy.deepcopy(x)
            self.lat = copy.deepcopy(y)
            self.rel.update(self.ocedata.find_rel_h(self.lon, self.lat))
        else:
            self.rel.update(HRel._make([None for i in range(11)]))
            self.lon = None
            self.lat = None
        if z is not None:
            self.dep = copy.deepcopy(z)
            if self.ocedata.readiness["Z"]:
                self.rel.update(self.ocedata.find_rel_v(z))
            else:
                self.rel.update(VRel._make(None for i in range(4)))
            if self.ocedata.readiness["Zl"]:
                self.rel.update(self.ocedata.find_rel_vl(z))
            else:
                self.rel.update(VlRel._make(None for i in range(4)))
        else:
            self.rel.update(VRel._make(None for i in range(4)))
            self.rel.update(VlRel._make(None for i in range(4)))
            self.dep = None

        if t is not None:
            self.t = copy.deepcopy(t)
            if self.ocedata.readiness["time"]:
                self.rel.update(self.ocedata.find_rel_t(t))
            else:
                self.rel.update(TRel._make(None for i in range(4)))
        else:
            self.rel.update(TRel._make(None for i in range(4)))
            self.t = None
            # (self.it, self.rt, self.dt, self.bt, self.t) = (None for i in range(5))
        return self

    def subset(self, which):
        """Create a subset of the Position object.

        Parameters
        ----------
        which: numpy.ndarray, optional
            Define which points survive the subset operation.
            It be an array of either boolean or int.
            The selection is similar to that of selecting from a 1D numpy array.

        Returns
        -------
        the_subset: Position object
            The selected Positions.
        """
        p = object.__new__(type(self))
        vardict = vars(self)
        keys = vardict.keys()
        for key in keys:
            item = vardict[key]
            if isinstance(item, np.ndarray):
                if len(item.shape) == 1:
                    setattr(p, key, item[which])
                    p.N = len(getattr(p, key))
                elif key in ["px", "py"]:
                    setattr(p, key, item[:, which])
                else:
                    setattr(p, key, item)
            elif isinstance(item, RelCoord):
                setattr(p, key, item.subset(which))
            else:
                setattr(p, key, item)
        return p

    def update_from_subset(self, sub, which):
        vardict = vars(sub)
        keys = vardict.keys()
        for key in keys:
            item = vardict[key]
            if not hasattr(self, key):
                logging.warning(
                    f"A new attribute {key} defined" "after updating from subset"
                )
                setattr(self, key, item)
            if getattr(self, key) is None:
                continue
            if isinstance(item, np.ndarray):
                if len(item.shape) == 1:
                    getattr(self, key)[which] = item
                elif key in ["px", "py"]:
                    getattr(self, key)[:, which] = item
            elif isinstance(item, RelCoord):
                self.rel.update_from_subset(item, which)
            elif isinstance(item, list):
                setattr(self, key, item)

    def _fatten_h(self, knw, ind_moves_kwarg={}):
        """Fatten horizontal indices.

        Fatten means to find the neighboring points of the points of interest based on the kernel.
        faces,iys,ixs are 1d arrays of size n.
        We are applying a kernel of size m.
        This is going to return a n * m array of indexes.
        A very slim vector is now a matrix, and hence the name.
        each row represen all the node needed for interpolation of a single point.
        "h" represent we are only doing it on the horizontal plane.

        Parameters
        ----------
        knw: KnW object
            The kernel used to find neighboring points.
        ind_moves_kward: dict, optional
            Key word argument to put into ind_moves method of the Topology object.
            Read Topology.ind_moves for more detail.
        """
        #         self.ind_h_dict
        kernel = knw.kernel.astype(int)
        kernel_tends = [_translate_to_tendency(k) for k in kernel]
        m = len(kernel_tends)
        n = len(self.iy)
        tp = self.ocedata.tp

        # the arrays we are going to return
        if self.face is not None:
            n_faces = np.zeros((n, m), int)
            n_faces.T[:] = self.face
        n_iys = np.zeros((n, m), int)
        n_ixs = np.zeros((n, m), int)

        # first try to fatten it naively(fast and vectorized)
        for i, node in enumerate(kernel):
            x_disp, y_disp = node
            n_iys[:, i] = self.iy + y_disp
            n_ixs[:, i] = self.ix + x_disp
        if self.face is not None:
            illegal = tp.check_illegal((n_faces, n_iys, n_ixs))
        else:
            illegal = tp.check_illegal((n_iys, n_ixs))

        redo = np.array(np.where(illegal)).T
        for loc in redo:
            j, i = loc
            if self.face is not None:
                ind = (self.face[j], self.iy[j], self.ix[j])
            else:
                ind = (self.iy[j], self.ix[j])
            # everyone start from the [0,0] node
            moves = kernel_tends[i]
            # moves is a list of operations to get to a single point
            # [2,2] means move to the left and then move to the left again.
            n_ind = tp.ind_moves(ind, moves, **ind_moves_kwarg)
            if self.face is not None:
                n_faces[j, i], n_iys[j, i], n_ixs[j, i] = n_ind
            else:
                n_iys[j, i], n_ixs[j, i] = n_ind
        if self.face is not None:
            return n_faces, n_iys, n_ixs
        else:
            return None, n_iys, n_ixs

    def _fatten_v(self, knw):
        """Fatten in vertical center coord.

        Find the neighboring center grid points in the vertical direction.

        Parameters
        ----------
        knw: KnW object
            The kernel used to find neighboring points.
        """
        if self.iz is None:
            return None
        if knw.vkernel == "nearest":
            return copy.deepcopy(self.iz)
        elif knw.vkernel in ["dz", "linear"]:
            try:
                self.iz_lin
            except AttributeError:
                self.rel.update(self.ocedata.find_rel_v_lin(self.dep))
            return np.vstack([self.iz_lin, self.iz_lin - 1]).T
        else:
            raise ValueError("vkernel not supported")

    def _fatten_vl(self, knw):
        """Fatten in vertical staggered coord.

        Finding the neighboring staggered grid points in the vertical direction.

        Parameters
        ----------
        knw: KnW object
            The kernel used to find neighboring points.
        """
        if self.izl is None:
            return None
        if knw.vkernel == "nearest":
            return copy.deepcopy(self.izl)
        elif knw.vkernel in ["dz", "linear"]:
            try:
                self.izl_lin
            except AttributeError:
                self.rel.update(self.ocedata.find_rel_vl_lin(self.dep))
            return np.vstack([self.izl_lin, self.izl_lin - 1]).T
        else:
            raise ValueError("vkernel not supported")

    def _fatten_t(self, knw):
        """Fatten in the temporal coord.

        Finding the neighboring center grid points in the temporal dimension.

        Parameters
        ----------
        knw: KnW object
            The kernel used to find neighboring points.
        """
        if self.it is None:
            return None
        if knw.tkernel == "nearest":
            return copy.deepcopy(self.it)
        elif knw.tkernel in ["dt", "linear"]:
            try:
                self.izl_lin
            except AttributeError:
                self.rel.update(self.ocedata.find_rel_t_lin(self.t))
            return np.vstack([self.it_lin, self.it_lin + 1]).T
        else:
            raise ValueError("tkernel not supported")

    def fatten(self, knw, four_d=False, required="all", ind_moves_kwarg={}):
        """Fatten in all the required dimensions.

        Finding the neighboring center grid points in all 4 dimensions.

        Parameters
        ----------
        knw: KnW object
            The kernel used to find neighboring points.
        four_d: Boolean, default False
            When we are doing nearest neighbor interpolation on some of the dimensions,
            with four_d = True, this will create dimensions with length 1, and will squeeze
            the dimension if four_d = False
        required: str, iterable of str, default "all"
            Which dims is needed in the process
        ind_moves_kward: dict, optional
            Key word argument to put into ind_moves method of the Topology object.
            Read Topology.ind_moves for more detail.
        """
        if required != "all" and isinstance(required, str):
            required = tuple([required])
        if required == "all" or isinstance(required, tuple):
            pass
        else:
            required = tuple(required)

        # TODO: register the kernel shape
        if (
            _in_required("X", required)
            or _in_required("Y", required)
            or _in_required("face", required)
        ):
            ffc, fiy, fix = self._fatten_h(knw, ind_moves_kwarg=ind_moves_kwarg)
            if ffc is not None:
                to_return = (ffc, fiy, fix)
                keys = ["face", "Y", "X"]
            else:
                to_return = (fiy, fix)
                keys = ["Y", "X"]
        else:
            to_return = tuple([np.zeros(self.N)])
            keys = ["place_holder"]

        if _in_required("Z", required):
            fiz = self._fatten_v(knw)
            if fiz is not None:
                to_return = _ind_broadcast(fiz, to_return)
                keys.insert(0, "Z")
        elif _in_required("Zl", required):
            fizl = self._fatten_vl(knw)
            if fizl is not None:
                to_return = _ind_broadcast(fizl, to_return)
                keys.insert(0, "Zl")
        elif four_d:
            to_return = [
                np.expand_dims(to_return[i], axis=-1) for i in range(len(to_return))
            ]

        if _in_required("time", required):
            fit = self._fatten_t(knw)
            if fit is not None:
                to_return = _ind_broadcast(fit, to_return)
                keys.insert(0, "time")
        elif four_d:
            to_return = [
                np.expand_dims(to_return[i], axis=-1) for i in range(len(to_return))
            ]
        to_return = dict(zip(keys, to_return))
        if required == "all":
            required = [i for i in keys if i != "place_holder"]
        return tuple(to_return[i] for i in required)

    def get_px_py(self):
        """Get the nearest 4 corner points of the given point.

        Used for oceanparcel style horizontal interpolation.
        """
        if self.face is not None:
            ind = (
                self.face,
                self.iy,
                self.ix,
            )
        else:
            ind = (self.iy, self.ix)
        px, py = find_px_py(
            self.ocedata.XG,
            self.ocedata.YG,
            self.ocedata.tp,
            ind,
        )
        px = self.lon + to_180(px - self.lon)
        return px, py

    def get_f_node_weight(self):
        """Find weight for the corner points interpolation."""
        return weight_f_node(self.rx, self.ry)

    def _register_interpolation_input(
        self, var_name, knw, prefetched=None, prefetch_prefix=None
    ):
        """Register the input of interpolation function.

        Part of the interpolation function.
        Register the input of the interpolation function.
        For the input format, go to interpolation for more details.

        Returns
        -------
        output_format: dict
            Record information about the original var_name input.
            Provide the formatting information for output,
            such that it matches the input in a direct fashion.
        main_keys: list
            A list that defines each interpolation to be performed.
            The combination of variable name and kernel uniquely define such an operation.
        prefetch_dict: dict
            Lookup the prefetched the data and its index prefix using main_key
        main_dict: dict
            Lookup the raw information using main_key
        hash_index: dict
            Lookup the token that uniquely define its indexing operation using main_key.
            Different main_key could share the same token.
        hash_mask: dict
            Lookup the token that uniquely define its masking operation using main_key.
            Different main_key could share the same token.
        hash_read: dict
            Lookup the token that uniquely define its reading of the data using main_key.
            Different main_key could share the same token.
        hash_weight: dict
            Lookup the token that uniquely define its computation of the weight using main_key.
            Different main_key could share the same token.
        """
        # prefetch_dict = {var:(prefetched,prefetch_prefix)}
        # main_dict = {var:(var,kernel)}
        # hash_index = {var:hash(cuvwg,kernel_size)}
        # hash_read  = {var:hash(var,kernel_size)}
        # hash_weight= {var:hash(kernel,cuvwg)}
        output_format = {}
        if isinstance(var_name, (str, tuple)):
            var_name = [var_name]
            output_format["single"] = True
        elif isinstance(var_name, list):
            output_format["single"] = False
        else:
            raise ValueError(
                "var_name needs to be string, tuple, or a list of the above."
            )
        num_var = len(var_name)

        if isinstance(knw, KnW):
            knw = [knw for i in range(num_var)]
        if isinstance(knw, tuple):
            if len(knw) != 2:
                raise ValueError(
                    "When knw is a tuple, we assume it to be kernels for a horizontal vector,"
                    "thus, it has to have 2 elements"
                )
            knw = [knw for i in range(num_var)]
        elif isinstance(knw, list):
            if len(knw) != num_var:
                raise ValueError("Mismatch between the number of kernels and variables")
        elif isinstance(knw, dict):
            temp = []
            for var in var_name:
                a_knw = knw.get(var)
                if a_knw is None or not isinstance(a_knw, (tuple, KnW)):
                    raise ValueError(
                        f"Variable {var} does not have a proper corresponding kernel"
                    )
                else:
                    temp.append(a_knw)
            knw = temp
        else:
            raise ValueError(
                "knw needs to be a KnW object, or list/dictionaries containing that "
            )

        if isinstance(prefetched, np.ndarray):
            prefetched = [prefetched for i in range(num_var)]
        elif isinstance(prefetched, tuple):
            prefetched = [prefetched for i in range(num_var)]
        elif prefetched is None:
            prefetched = [prefetched for i in range(num_var)]
        elif isinstance(prefetched, list):
            if len(prefetched) != num_var:
                raise ValueError(
                    "Mismatch between the number of prefetched arrays and variables"
                )
        elif isinstance(prefetched, dict):
            prefetched = [prefetched.get(var) for var in var_name]
        else:
            raise ValueError(
                "prefetched needs to be a numpy array/tuple pair of numpy array,"
                " or list/dictionaries containing that"
            )

        if isinstance(prefetch_prefix, tuple):
            prefetch_prefix = [prefetch_prefix for i in range(num_var)]
        elif prefetch_prefix is None:
            prefetch_prefix = [None for i in range(num_var)]
        elif isinstance(prefetch_prefix, list):
            if len(prefetch_prefix) != num_var:
                raise ValueError(
                    "Mismatch between the number of prefetched arrays prefix prefetch_prefix and variables"
                )
        elif isinstance(prefetch_prefix, dict):
            prefetch_prefix = [prefetch_prefix.get(var) for var in var_name]
        else:
            raise ValueError(
                "prefetched prefix prefetch_prefix needs to be a tuple, or list/dictionaries containing that "
            )

        output_format["ori_list"] = copy.deepcopy(list(zip(var_name, knw)))
        new_var_name = []
        new_prefetched = []
        new_knw = []
        new_prefetch_prefix = []
        for i, var in enumerate(var_name):
            if isinstance(var, str):
                new_var_name.append(var)
                new_prefetched.append(prefetched[i])
                new_knw.append(knw[i])
                new_prefetch_prefix.append(prefetch_prefix[i])
            elif isinstance(var, tuple):
                if self.face is None:
                    for j in range(len(var)):
                        new_var_name.append(var[j])
                        if prefetched[i] is not None:
                            new_prefetched.append(prefetched[i][j])
                        else:
                            new_prefetched.append(None)
                        new_knw.append(knw[i][j])
                        new_prefetch_prefix.append(prefetch_prefix[i])
                else:
                    new_var_name.append(var)
                    new_prefetched.append(prefetched[i])
                    new_knw.append(knw[i])
                    new_prefetch_prefix.append(prefetch_prefix[i])
            elif var is None:
                pass
            else:
                raise ValueError(
                    "var_name needs to be string, tuple, or a list of the above."
                )

        prefetched = new_prefetched
        knw = new_knw
        prefetch_prefix = new_prefetch_prefix
        var_name = new_var_name
        output_format["final_var_name"] = list(zip(var_name, knw))

        kernel_size_hash = []
        kernel_hash = []
        mask_ignore = []
        for kkk in knw:
            if isinstance(kkk, KnW):
                kernel_size_hash.append(kkk.size_hash())
                kernel_hash.append(hash(kkk))
                mask_ignore.append(kkk.ignore_mask)
            elif isinstance(kkk, tuple):
                if len(kkk) != 2:
                    raise ValueError(
                        "When knw is a tuple, we assume it to be kernels for a horizontal vector,"
                        "thus, it has to have 2 elements"
                    )
                uknw, vknw = kkk
                kernel_size_hash.append(uknw.size_hash())
                kernel_hash.append(hash((uknw, vknw)))
                mask_ignore.append(uknw.ignore_mask or vknw.ignore_mask)
        dims = []
        for var in var_name:
            if isinstance(var, str):
                dims.append(self.ocedata[var].dims)
            elif isinstance(var, tuple):
                temp = []
                for vvv in var:
                    temp.append(self.ocedata[vvv].dims)
                dims.append(tuple(temp))

        main_keys = list(zip(var_name, kernel_hash))
        prefetch_dict = dict(zip(main_keys, zip(prefetched, prefetch_prefix)))
        main_dict = dict(zip(main_keys, zip(var_name, dims, knw)))
        hash_index = dict(
            zip(main_keys, [hash(i) for i in zip(dims, kernel_size_hash)])
        )
        hash_mask = dict(
            zip(main_keys, [hash(i) for i in zip(dims, mask_ignore, kernel_size_hash)])
        )
        hash_read = dict(
            zip(main_keys, [hash(i) for i in zip(var_name, kernel_size_hash)])
        )
        hash_weight = dict(zip(main_keys, [hash(i) for i in zip(dims, kernel_hash)]))
        return (
            output_format,
            main_keys,
            prefetch_dict,
            main_dict,
            hash_index,
            hash_mask,
            hash_read,
            hash_weight,
        )

    def _fatten_required_index_and_register(self, hash_index, main_dict):
        """Fatten for the interpolation process.

        Perform the fatten process for each unique token. Register the result as a dict.

        Parameters
        ----------
        hash_index: dict
            See _register_interpolation_input
        main_dict: dict
            See _register_interpolation_input

        Returns
        -------
        + index_lookup: dict
            A dictionary to lookup fatten results.
            The keys are the token in hash_index.
            The values are corresponding results.
        """
        hsh = np.unique(list(hash_index.values()))
        index_lookup = {}
        for hs in hsh:
            main_key = get_key_by_value(hash_index, hs)
            var_name, dims, knw = main_dict[main_key]
            if isinstance(var_name, str):
                old_dims = dims
            elif isinstance(var_name, tuple):
                old_dims = dims[0]
            dims = []
            for i in old_dims:
                if i in ["Xp1", "Yp1"]:
                    dims.append(i[:1])
                else:
                    dims.append(i)
            dims = tuple(dims)
            if isinstance(var_name, str):
                if "Xp1" in old_dims and "Yp1" in old_dims:
                    cuvwg = "G"
                else:
                    cuvwg = "C"
                ind = self.fatten(
                    knw, required=dims, four_d=True, ind_moves_kwarg={"cuvwg": cuvwg}
                )
                index_lookup[hs] = ind
            elif isinstance(var_name, tuple):
                uknw, vknw = knw
                uind = self.fatten(
                    uknw, required=dims, four_d=True, ind_moves_kwarg={"cuvwg": "U"}
                )
                vind = self.fatten(
                    vknw, required=dims, four_d=True, ind_moves_kwarg={"cuvwg": "V"}
                )
                index_lookup[hs] = (uind, vind)

        return index_lookup

    def _transform_vector_and_register(self, index_lookup, hash_index, main_dict):
        """Transform vectors for interpolation.

        Perform the vector transformation.
        This is to say that sometimes u velocity becomes v velocity for datasets with face topology.
          Register the result as a dict.

        Parameters
        ----------
        index_lookup: dict
            See _fatten_required_index_and_register
        hash_index: dict
            See _register_interpolation_input
        main_dict: dict
            See _register_interpolation_input

        Returns
        -------
        + transform_lookup: dict
            A dictionary to lookup transformation results.
            The keys are the token in hash_index.
            The values are corresponding results.
        """
        hsh = np.unique(list(hash_index.values()))
        transform_lookup = {}
        if self.face is None:
            for hs in hsh:
                transform_lookup[hs] = None
            return transform_lookup
        for hs in hsh:
            main_key = get_key_by_value(hash_index, hs)
            var_name, dims, knw = main_dict[main_key]
            if isinstance(var_name, str):
                transform_lookup[hs] = None
            elif isinstance(var_name, tuple):
                uind, vind = index_lookup[hs]
                uind_dic = dict(zip(dims[0], uind))
                vind_dic = dict(zip(dims[1], vind))
                # This only matters when dim == 'face', no need to think about 'Xp1'
                (UfromUvel, UfromVvel, _, _) = self.ocedata.tp.four_matrix_for_uv(
                    uind_dic["face"][:, :, 0, 0]
                )

                (_, _, VfromUvel, VfromVvel) = self.ocedata.tp.four_matrix_for_uv(
                    vind_dic["face"][:, :, 0, 0]
                )

                UfromUvel = np.round(UfromUvel)
                UfromVvel = np.round(UfromVvel)
                VfromUvel = np.round(VfromUvel)
                VfromVvel = np.round(VfromVvel)

                bool_ufromu = np.abs(UfromUvel).astype(bool)
                bool_ufromv = np.abs(UfromVvel).astype(bool)
                bool_vfromu = np.abs(VfromUvel).astype(bool)
                bool_vfromv = np.abs(VfromVvel).astype(bool)

                indufromu = tuple(idid[bool_ufromu] for idid in uind)
                indufromv = tuple(idid[bool_ufromv] for idid in uind)
                indvfromu = tuple(idid[bool_vfromu] for idid in vind)
                indvfromv = tuple(idid[bool_vfromv] for idid in vind)

                transform_lookup[hs] = (
                    (UfromUvel, UfromVvel, VfromUvel, VfromVvel),
                    (bool_ufromu, bool_ufromv, bool_vfromu, bool_vfromv),
                    (indufromu, indufromv, indvfromu, indvfromv),
                )
            else:
                raise ValueError(f"unsupported dims: {dims}")
        # modify the index_lookup
        return transform_lookup

    def _mask_value_and_register(
        self, index_lookup, transform_lookup, hash_mask, hash_index, main_dict
    ):
        """Create masks for interpolation.

        Perform the masking process and register in a dictionary.

        Parameters
        ----------
        index_lookup: dict
            See _fatten_required_index_and_register
        transform_lookup: dict
            See _transform_vector_and_lookup
        hash_mask: dict
            See _register_interpolation_input
        hash_index: dict
            See _register_interpolation_input
        main_dict: dict
            See _register_interpolation_input

        Returns
        -------
        + mask_lookup: dict
            A dictionary to lookup masking results.
            The keys are the token in hash_mask.
            The values are corresponding results.
        """
        hsh = np.unique(list(hash_mask.values()))
        mask_lookup = {}
        for hs in hsh:
            main_key = get_key_by_value(hash_mask, hs)
            var_name, dims, knw = main_dict[main_key]
            hsind = hash_index[main_key]
            longDims = "".join([str(a_thing) for a_thing in dims])
            if isinstance(knw, KnW):
                ignore_mask = knw.ignore_mask
            elif isinstance(knw, tuple):
                ignore_mask = (knw[0].ignore_mask) or (knw[1].ignore_mask)

            if ignore_mask or ("X" not in longDims) or ("Y" not in longDims):
                mask_lookup[hs] = None
            elif isinstance(var_name, str):
                ind = index_lookup[hsind]
                ind_for_mask = _ind_for_mask(ind, dims)
                if "Zl" in dims:
                    cuvw = "Wvel"
                elif "Z" in dims:
                    if "Xp1" in dims and "Yp1" in dims:
                        raise NotImplementedError(
                            "The masking of corner points are open to "
                            "interpretations thus not implemented, "
                            "let knw.ignore_mask =True to go around"
                        )
                    elif "Xp1" in dims:
                        cuvw = "U"
                    elif "Yp1" in dims:
                        cuvw = "V"
                    else:
                        cuvw = "C"
                else:
                    cuvw = "C"
                masked = get_masked(self.ocedata, ind_for_mask, cuvwg=cuvw)
                mask_lookup[hs] = masked
            elif isinstance(var_name, tuple):
                to_unzip = transform_lookup[hsind]
                uind, vind = index_lookup[hsind]
                if to_unzip is None:
                    uind_for_mask = _ind_for_mask(uind, dims[0])
                    vind_for_mask = _ind_for_mask(vind, dims[1])
                    umask = get_masked(self.ocedata, uind_for_mask, cuvwg="U")
                    vmask = get_masked(self.ocedata, vind_for_mask, cuvwg="V")
                else:
                    (
                        _,
                        (bool_ufromu, bool_ufromv, bool_vfromu, bool_vfromv),
                        (indufromu, indufromv, indvfromu, indvfromv),
                    ) = to_unzip
                    umask = np.full(uind[0].shape, np.nan)
                    vmask = np.full(vind[0].shape, np.nan)

                    umask[bool_ufromu] = get_masked(
                        self.ocedata, _ind_for_mask(indufromu, dims[0]), cuvwg="U"
                    )
                    umask[bool_ufromv] = get_masked(
                        self.ocedata, _ind_for_mask(indufromv, dims[1]), cuvwg="V"
                    )
                    vmask[bool_vfromu] = get_masked(
                        self.ocedata, _ind_for_mask(indvfromu, dims[0]), cuvwg="U"
                    )
                    vmask[bool_vfromv] = get_masked(
                        self.ocedata, _ind_for_mask(indvfromv, dims[1]), cuvwg="V"
                    )
                mask_lookup[hs] = (umask, vmask)
        return mask_lookup

    def _read_data_and_register(
        self,
        index_lookup,
        transform_lookup,
        hash_read,
        hash_index,
        main_dict,
        prefetch_dict,
    ):
        """Read the data and register them as dict.

        Parameters
        ----------
        index_lookup: dict
            See _fatten_required_index_and_register
        transform_lookup: dict
            See _transform_vector_and_lookup
        hash_read: dict
            See _register_interpolation_input
        hash_index: dict
            See _register_interpolation_input
        main_dict: dict
            See _register_interpolation_input
        prefetch_dict: dict
            See _register_interpolation_input

        Returns
        -------
        + read_lookup: dict
            A dictionary to lookup data reading results.
            The keys are the token in hash_read.
            The values are corresponding results.
        """
        hsh = np.unique(list(hash_read.values()))
        data_lookup = {}
        for hs in hsh:
            main_key = get_key_by_value(hash_read, hs)
            hsind = hash_index[main_key]
            var_name, dims, knw = main_dict[main_key]
            prefetched, prefetch_prefix = prefetch_dict[main_key]
            if isinstance(var_name, str):
                ind = index_lookup[hsind]
                if prefetched is not None:
                    if prefetch_prefix is None:
                        raise ValueError(
                            "please pass value of the prefix of prefetched dataset, "
                            "even if the prefix is zero"
                        )
                    temp_ind = _subtract_prefetch_prefix(ind, prefetch_prefix)
                    needed = np.nan_to_num(prefetched[temp_ind])
                else:
                    needed = np.nan_to_num(smart_read(self.ocedata[var_name], ind))
                data_lookup[hs] = needed
            elif isinstance(var_name, tuple):
                uname, vname = var_name
                uind, vind = index_lookup[hsind]
                (
                    (UfromUvel, UfromVvel, VfromUvel, VfromVvel),
                    (bool_ufromu, bool_ufromv, bool_vfromu, bool_vfromv),
                    (indufromu, indufromv, indvfromu, indvfromv),
                ) = transform_lookup[hsind]
                temp_n_u = np.full(uind[0].shape, np.nan)
                temp_n_v = np.full(vind[0].shape, np.nan)
                if prefetched is not None:
                    upre, vpre = prefetched
                    ufromu = np.nan_to_num(
                        upre[_subtract_prefetch_prefix(indufromu, prefetch_prefix)]
                    )
                    ufromv = np.nan_to_num(
                        vpre[_subtract_prefetch_prefix(indufromv, prefetch_prefix)]
                    )
                    vfromu = np.nan_to_num(
                        upre[_subtract_prefetch_prefix(indvfromu, prefetch_prefix)]
                    )
                    vfromv = np.nan_to_num(
                        vpre[_subtract_prefetch_prefix(indvfromv, prefetch_prefix)]
                    )
                else:
                    ufromu = np.nan_to_num(smart_read(self.ocedata[uname], indufromu))
                    ufromv = np.nan_to_num(smart_read(self.ocedata[vname], indufromv))
                    vfromu = np.nan_to_num(smart_read(self.ocedata[uname], indvfromu))
                    vfromv = np.nan_to_num(smart_read(self.ocedata[vname], indvfromv))
                temp_n_u[bool_ufromu] = ufromu  # 0#
                temp_n_u[bool_ufromv] = ufromv  # 1#
                temp_n_v[bool_vfromu] = vfromu  # 0#
                temp_n_v[bool_vfromv] = vfromv  # 1#

                n_u = np.einsum("nijk,ni->nijk", temp_n_u, UfromUvel) + np.einsum(
                    "nijk,ni->nijk", temp_n_u, UfromVvel
                )
                n_v = np.einsum("nijk,ni->nijk", temp_n_v, VfromUvel) + np.einsum(
                    "nijk,ni->nijk", temp_n_v, VfromVvel
                )
                data_lookup[hs] = (n_u, n_v)

        return data_lookup

    def _compute_weight_and_register(
        self, mask_lookup, hash_weight, hash_mask, main_dict
    ):
        """Compute the weights and register them as dict.

        Parameters
        ----------
        mask_lookup: dict
            See _mask_value_and_register
        hash_weight: dict
            See _register_interpolation_input
        hash_mask: dict
            See _register_interpolation_input
        main_dict: dict
            See _register_interpolation_input

        Returns
        -------
        + weight_lookup: dict
            A dictionary to lookup the weights computed.
            The keys are the token in hash_weight.
            The values are corresponding results.
        """
        hsh = np.unique(list(hash_weight.values()))
        weight_lookup = {}
        for hs in hsh:
            main_key = get_key_by_value(hash_weight, hs)
            hsmsk = hash_mask[main_key]
            var_name, dims, knw = main_dict[main_key]
            masked = mask_lookup[hsmsk]
            if isinstance(var_name, tuple):
                ori_dims = dims
                dims = ori_dims[0]
                ori_knw = knw
                knw = ori_knw[0]
                # Assuming the two kernels have the same
                # vertical dimensions, which is reasonable.

            # shared part for 'vertical direction'
            this_bottom_scheme = "no_flux"
            if "Z" in dims:
                if self.rz is not None:
                    if knw.vkernel == "nearest":
                        rz = self.rz
                    else:
                        rz = self.rz_lin
                else:
                    rz = 0
            elif "Zl" in dims:
                this_bottom_scheme = None
                if self.rzl is not None:
                    if knw.vkernel == "nearest":
                        rz = self.rzl
                    else:
                        rz = self.rzl_lin
                else:
                    rz = 0
            else:
                rz = 0
            if self.rt is not None:
                if knw.tkernel == "nearest":
                    rt = self.rt
                else:
                    rt = self.rt_lin
            else:
                rt = 0

            if isinstance(var_name, str):
                if "Xp1" in dims:
                    rx = self.rx + 0.5
                else:
                    rx = self.rx
                if "Yp1" in dims:
                    ry = self.ry + 0.5
                else:
                    ry = self.ry
                if masked is None:
                    pk4d = None
                else:
                    pk4d = _find_pk_4d(masked, inheritance=knw.inheritance)
                weight = knw.get_weight(
                    rx=rx,
                    ry=ry,
                    rz=rz,
                    rt=rt,
                    pk4d=pk4d,
                    bottom_scheme=this_bottom_scheme,
                )
                weight_lookup[hs] = weight
            elif isinstance(var_name, tuple):
                uknw, vknw = ori_knw
                if masked is None:
                    upk4d = None
                    vpk4d = None
                else:
                    umask, vmask = masked
                    upk4d = _find_pk_4d(umask, inheritance=uknw.inheritance)
                    vpk4d = _find_pk_4d(vmask, inheritance=vknw.inheritance)
                uweight = uknw.get_weight(
                    self.rx + 1 / 2, self.ry, rz=rz, rt=rt, pk4d=upk4d
                )
                vweight = vknw.get_weight(
                    self.rx, self.ry + 1 / 2, rz=rz, rt=rt, pk4d=vpk4d
                )
                weight_lookup[hs] = (uweight, vweight)
        return weight_lookup

    def interpolate(
        self, var_name, knw, vec_transform=True, prefetched=None, prefetch_prefix=None
    ):
        """Do interpolation.

        This is the method that does the actual interpolation/derivative.
        It is a combination of the following methods:
        _register_interpolation_input,
        _fatten_required_index_and_register,
        _transform_vector_and_register,
        _read_data_and_register,
        _mask_value_and_register,
        _compute_weight_and_registe,.

        Parameters
        ----------
        var_name: list, str, tuple
            The variables to interpolate. Tuples are used for horizontal vectors.
            Put str and list in a list if you have multiple things to interpolate.
            This input also defines the format of the output.
        knw: KnW object, tuple, list, dict
            The kernel object used for the operation.
            Put them in the same order as var_name.
            Some level of automatic broadcasting is also supported.
        vec_transform: Boolean
            Whether to project the vector field to the local zonal/meridional direction.
        prefetched: numpy.ndarray, tuple, list, dict, None, default None
            The prefetched array for the data, this will effectively overwrite var_name.
            Put them in the same order as var_name.
            Some level of automatic broadcasting is also supported.
        prefetch_prefix: tuple, list, dict, None, default None
            The prefix of the prefetched array.
            Put them in the same order as var_name.
            Some level of automatic broadcasting is also supported.

        Returns
        -------
        to_return: list, numpy.array, tuple
            The interpolation/derivative output in the same format as var_name.
        """
        to_return = []
        (
            output_format,
            main_keys,
            prefetch_dict,
            main_dict,
            hash_index,
            hash_mask,
            hash_read,
            hash_weight,
        ) = self._register_interpolation_input(
            var_name, knw, prefetched=prefetched, prefetch_prefix=prefetch_prefix
        )
        index_lookup = self._fatten_required_index_and_register(hash_index, main_dict)
        transform_lookup = self._transform_vector_and_register(
            index_lookup, hash_index, main_dict
        )
        data_lookup = self._read_data_and_register(
            index_lookup,
            transform_lookup,
            hash_read,
            hash_index,
            main_dict,
            prefetch_dict,
        )
        mask_lookup = self._mask_value_and_register(
            index_lookup, transform_lookup, hash_mask, hash_index, main_dict
        )
        weight_lookup = self._compute_weight_and_register(
            mask_lookup, hash_weight, hash_mask, main_dict
        )
        # index_list = []
        for key in main_keys:
            var_name, dims, knw = main_dict[key]
            if isinstance(var_name, str):
                needed = data_lookup[hash_read[key]]
                weight = weight_lookup[hash_weight[key]]
                needed = _partial_flatten(needed)
                weight = _partial_flatten(weight)
                to_return.append(np.einsum("nj,nj->n", needed, weight))
                # index_list.append((index_lookup[hash_index[key]],
                #                    transform_lookup[hash_index[key]],
                #                    data_lookup[hash_read[key]]))
            elif isinstance(var_name, tuple):
                n_u, n_v = data_lookup[hash_read[key]]
                uweight, vweight = weight_lookup[hash_weight[key]]
                u = np.einsum("nijk,nijk->n", n_u, uweight)
                v = np.einsum("nijk,nijk->n", n_v, vweight)
                if vec_transform:
                    u, v = local_to_latlon(u, v, self.cs, self.sn)
                to_return.append((u, v))
                # index_list.append((index_lookup[hash_index[key]],
                #                    transform_lookup[hash_index[key]],
                #                    data_lookup[hash_read[key]]))
            else:
                raise ValueError(f"unexpected var_name: {var_name}")

        final_dict = dict(zip(output_format["final_var_name"], to_return))
        ori_list = output_format["ori_list"]
        output = []
        # print(ori_list,to_return,final_dict.keys())
        for key in ori_list:
            var, knw = key

            if var is None:
                output.append(None)
            elif isinstance(var, tuple):
                if self.face is None:
                    temp_key = [(var[i], knw[i]) for i in range(len(var))]
                    values = tuple(final_dict.get(k) for k in temp_key)
                    output.append(values)
                else:
                    output.append(final_dict.get(key))
            else:
                output.append(final_dict.get(key))
        if output_format["single"]:
            output = output[0]
        return output
