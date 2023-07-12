import copy
import logging
from functools import cache
from itertools import combinations

import numpy as np

from seaduck.runtime_conf import compileable

# default kernel for interpolation.
DEFAULT_KERNEL = np.array(
    [[0, 0], [0, 1], [0, 2], [0, -1], [0, -2], [-1, 0], [-2, 0], [1, 0], [2, 0]]
)
DEFAULT_INHERITANCE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 5, 7, 8],
    [0, 1, 3, 5, 7],
    [0],
]

DEFAULT_KERNELS = [
    np.array([DEFAULT_KERNEL[i] for i in doll]) for doll in DEFAULT_INHERITANCE
]


# It just tell you what the kernels look like
def show_kernels(kernels=DEFAULT_KERNELS):
    """Plot a small scatter plot of the shape of a list of kernel.

    Parameters
    ----------
    kernels: list of numpy.ndarray
        Each of the element is a (n,2) shaped array,
        where n is the number of element in the kernel.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("maptlotlib.pyplot is needed to use this function.") from exc

    for i, k in enumerate(kernels):
        x, y = k.T
        plt.plot(x + 0.1 * i, y + 0.1 * i, "+")


def _translate_to_tendency(kernel):
    """Translate a movement to directions.

    A kernel looks like
    np.array([
    [x1,y1],
    [x2,y2],....
    ])
    where [xn,yn] represent a coordinate relative to [0,0]
    which is just the nearest neighbor.
    this function return how you could move from [0,0] to k = [x,y]
    If you need to go to [0,2], you need to move up twice.
    it will return [0,0] or more explicitly ['up','up']
    [0,0] will produce a empty array.

    Parameters
    ----------
    k: numpy.ndarray
        A (n,2)-shaped array, where n is the number of
        element in the kernel.
    """
    tend = []
    x, y = kernel
    if y > 0:
        for j in range(y):
            tend.append(0)  # up
    else:
        for j in range(-y):
            tend.append(1)  # down
    if x < 0:
        for j in range(-x):
            tend.append(2)  # left
    else:
        for j in range(x):
            tend.append(3)  # right
    return tend


def kernel_weight_x(kernel, kernel_type="interp", order=0):
    r"""Return the function that calculate the interpolation/derivative weight.

    input needs to be a cross-shaped (that's where x is coming from) Lagrangian kernel.

    If you don't want to know what is going on under the hood.
    it's totally fine.

    all of the following is a bit hard to understand.
    The k th (k>=0) derivative of the lagrangian polynomial is
    $$
    w_j= \frac{\Sigma_{i\neq j} Pi_{i<m-1-k} (x-x_i)}{\Pi_{i\neq j} (x_j - x_i)}
    $$
    for example: if the points are [-1,0,1] for point 0
    k = 0: w = (x-1)(x+1)/(0-1)(0+1)
    k = 1: w = [(x+1)+(x-1)]/(0-1)(0+1)

    for a cross shape kernel:
    f(rx,ry) = f_x(rx) + f_y(ry) - f(0,0)

    Parameters
    ----------
    kernel: np.ndarray
        2D array with shape (n,2), where n is the number of nodes.
        It has to be shaped like a cross
    kernel_type: str
        "interp" (default): Use both x y direction for interpolation,
        implies that order = 0
        "x": Use only x direction for interpolation/derivative
        "y": Use only y direction for interpolation/derivative
    order: int
        The order of derivatives. Zero for interpolation.

    Returns
    -------
    func(rx,ry): compilable function
        function to calculate the hotizontal interpolation/derivative
        weight
    """
    xs = np.array(list(set(kernel.T[0])), dtype=float)
    ys = np.array(list(set(kernel.T[1])), dtype=float)

    # if you the kernel is a line rather than a cross
    if len(xs) == 1:
        kernel_type = "y"
    elif len(ys) == 1:
        kernel_type = "x"

    x_poly = []
    y_poly = []
    if kernel_type == "interp":
        for ax in xs:
            x_poly.append(list(combinations([i for i in xs if i != ax], len(xs) - 1)))
        for ay in ys:
            y_poly.append(list(combinations([i for i in ys if i != ay], len(ys) - 1)))
    if kernel_type == "x":
        for ax in xs:
            x_poly.append(
                list(combinations([i for i in xs if i != ax], len(xs) - 1 - order))
            )
        y_poly = [[[]]]
    if kernel_type == "y":
        x_poly = [[[]]]
        for ay in ys:
            y_poly.append(
                list(combinations([i for i in ys if i != ay], len(ys) - 1 - order))
            )
    x_poly = np.array(x_poly, dtype=float)
    y_poly = np.array(y_poly, dtype=float)

    @compileable
    def the_interp_func(rx, ry):
        nonlocal kernel, xs, ys, x_poly, y_poly
        n = len(rx)
        m = len(kernel)
        weight = np.ones((n, m)) * 0.0
        for i, (x, y) in enumerate(kernel):
            if x != 0:
                ix = np.where(xs == x)[0][0]
                poly = x_poly[ix]
                for term in poly:
                    another = np.ones(n)
                    for other in term:
                        another *= rx - other
                    weight[:, i] += another
                    for other in xs:
                        if other != x:
                            weight[:, i] /= x - other
            if y != 0:
                iy = np.where(ys == y)[0][0]
                poly = y_poly[iy]
                for term in poly:
                    another = np.ones(n) * 1.0
                    for other in term:
                        another *= ry - other
                    weight[:, i] += another
                    for other in ys:
                        if other != y:
                            weight[:, i] /= y - other
            elif x**2 + y**2 == 0:
                xthing = np.zeros(n) * 0.0
                ix = np.where(xs == 0)[0][0]
                poly = x_poly[ix]
                for term in poly:
                    another = np.ones(n) * 1.0
                    for other in term:
                        another *= rx - other
                    xthing += another
                    for other in xs:
                        if other != x:
                            xthing /= x - other

                ything = np.zeros(n) * 0.0
                iy = np.where(ys == y)[0][0]
                poly = y_poly[iy]
                for term in poly:
                    another = np.ones(n)
                    for other in term:
                        another *= ry - other
                    ything += another
                    for other in ys:
                        if other != y:
                            ything /= y - other
                weight[:, i] = xthing + ything - 1
        return weight

    @compileable
    def the_x_func(rx, ry):
        nonlocal kernel, xs, ys, x_poly, order
        n = len(rx)
        m = len(kernel)
        weight = np.ones((n, m)) * 0.0
        for i, (x, y) in enumerate(kernel):
            if y == 0:
                ix = np.where(xs == x)[0][0]
                poly = x_poly[ix]
                for term in poly:
                    another = np.ones(n) * 1.0
                    for other in term:
                        another *= rx - other

                    weight[:, i] += another
                for other in xs:
                    if other != x:
                        weight[:, i] /= x - other
        return weight

    @compileable
    def the_x_maxorder_func(rx, ry):
        nonlocal kernel, xs, ys, order
        n = len(rx)
        m = len(kernel)
        common = 1
        for i in range(1, order):
            common *= i
        weight = np.ones((n, m)) * float(common)
        for i, (x, y) in enumerate(kernel):
            if y == 0:
                for other in xs:
                    if other != x:
                        weight[:, i] /= x - other
            else:
                weight[:, i] = 0.0
        return weight

    @compileable
    def the_y_func(rx, ry):
        nonlocal kernel, xs, ys, y_poly, order
        n = len(rx)
        m = len(kernel)
        weight = np.ones((n, m)) * 0.0
        for i, (x, y) in enumerate(kernel):
            if x == 0:
                iy = np.where(ys == y)[0][0]
                poly = y_poly[iy]
                for term in poly:
                    another = np.ones(n) * 1.0
                    for other in term:
                        another *= ry - other

                    weight[:, i] += another
                for other in ys:
                    if other != y:
                        weight[:, i] /= y - other
        return weight

    @compileable
    def the_y_maxorder_func(rx, ry):
        nonlocal kernel, xs, ys, order
        n = len(rx)
        m = len(kernel)
        common = 1
        for i in range(1, order):
            common *= i
        weight = np.ones((n, m)) * float(common)
        for i, (x, y) in enumerate(kernel):
            if x == 0:
                for other in ys:
                    if other != y:
                        weight[:, i] /= y - other
            else:
                weight[:, i] = 0.0
        return weight

    if kernel_type == "interp":
        return the_interp_func
    if kernel_type == "x":
        max_order = len(xs) - 1
        if order < max_order:
            return the_x_func
        elif order == max_order:
            return the_x_maxorder_func
        else:
            raise ValueError("Kernel is too small for this derivative")
    if kernel_type == "y":
        max_order = len(ys) - 1
        if order < max_order:
            return the_y_func
        elif order == max_order:
            return the_y_maxorder_func
        else:
            raise ValueError("Kernel is too small for this derivative")


# we can define the default interpolation functions here,
# so if we are using it over and over, we don't have to compile it.
# and it really takes a lot of time to compile.


def kernel_weight_s(kernel, xorder=0, yorder=0):
    """Return the function that calculate the interpolation/derivative weight.

    input needs to be a rectangle-shaped (that's where x is coming from)
    Lagrangian kernel.

    Parameters
    ----------
    kernel: np.ndarray
        2D array with shape (n,2), where n is the number of nodes.
        It has to be shaped like a rectangle
    xorder: int
        The order of derivatives in the x direction.
        Zero for interpolation.
    yorder: int
        The order of derivatives in the y direction.
        Zero for interpolation.

    Returns
    -------
    func(rx,ry): compilable function
        function to calculate the hotizontal interpolation/derivative
        weight
    """
    xs = np.array(list(set(kernel.T[0])), dtype=float)
    ys = np.array(list(set(kernel.T[1])), dtype=float)
    xmaxorder = False
    ymaxorder = False
    if xorder < len(xs) - 1:
        pass
    elif xorder == len(xs) - 1:
        xmaxorder = True
    else:
        raise ValueError("Kernel is too small for this derivative")

    if yorder < len(ys) - 1:
        pass
    elif yorder == len(ys) - 1:
        ymaxorder = True
    else:
        raise ValueError("Kernel is too small for this derivative")

    x_poly = []
    y_poly = []
    for ax in xs:
        x_poly.append(
            list(combinations([i for i in xs if i != ax], len(xs) - 1 - xorder))
        )
    for ay in ys:
        y_poly.append(
            list(combinations([i for i in ys if i != ay], len(ys) - 1 - yorder))
        )
    x_poly = np.array(x_poly, dtype=float)
    y_poly = np.array(y_poly, dtype=float)

    @compileable
    def the_square_func(rx, ry):
        nonlocal kernel, xs, ys, y_poly, x_poly, xorder, yorder, xmaxorder, ymaxorder
        n = len(rx)
        num_node_x = len(xs)
        num_node_y = len(ys)
        m = len(kernel)
        yweight = np.ones((n, num_node_y))
        xweight = np.ones((n, num_node_x))
        weight = np.ones((n, m)) * 0.0

        if ymaxorder:
            common = 1
            for i in range(1, yorder):
                common *= i
            yweight *= float(common)
        else:
            yweight *= 0.0
        for i, y in enumerate(ys):
            if not ymaxorder:
                iy = np.where(ys == y)[0][0]
                poly = y_poly[iy]
                for term in poly:
                    another = np.ones(n) * 1.0
                    for other in term:
                        another *= ry - other

                    yweight[:, i] += another
            for other in ys:
                if other != y:
                    yweight[:, i] /= y - other

        if xmaxorder:
            common = 1
            for i in range(1, xorder):
                common *= i
            xweight *= float(common)
        else:
            xweight *= 0.0
        for i, x in enumerate(xs):
            if not xmaxorder:
                ix = np.where(xs == x)[0][0]
                poly = x_poly[ix]
                for term in poly:
                    another = np.ones(n) * 1.0
                    for other in term:
                        another *= rx - other

                    xweight[:, i] += another
            for other in xs:
                if other != x:
                    xweight[:, i] /= x - other

        for i, (x, y) in enumerate(kernel):
            iy = np.where(ys == y)[0][0]
            ix = np.where(xs == x)[0][0]
            weight[:, i] = yweight[:, iy] * xweight[:, ix]

        return weight

    return the_square_func


def kernel_weight(kernel, kernel_type="interp", order=0):
    """Return a function that compute weights.

    A wrapper around kernel_weight_x and kernel_weight_s.
    Return the function that calculate the interpolation/derivative weight
    of a  Lagrangian kernel.

    Parameters
    ----------
    kernel: np.ndarray
        2D array with shape (n,2), where n is the number of nodes.
        It need to either shape like a rectangle or a cross
    kernel_type: str
        "interp" (default): Use both x y direction for interpolation,
        implies that order = 0
        "dx": Use only x direction for interpolation/derivative
        "dy": Use only y direction for interpolation/derivative
    order: int
        The order of derivatives. Zero for interpolation.

    Returns
    -------
    func(rx,ry): compilable function
        function to calculate the hotizontal interpolation/derivative
        weight
    """
    num_node_x = len(set(kernel[:, 0]))
    num_node_y = len(set(kernel[:, 1]))
    if len(kernel) == num_node_x + num_node_y - 1:
        if "d" in kernel_type:
            kernel_type = kernel_type[1:]
        return kernel_weight_x(kernel, kernel_type=kernel_type, order=order)
    elif len(kernel) == num_node_x * num_node_y:
        # num_node_x*num_node_y == num_node_x+num_node_y-1
        # only when num_node_x==1 or num_node_y ==1
        if kernel_type == "interp":
            return kernel_weight_s(kernel, xorder=0, yorder=0)
        elif kernel_type == "dx":
            return kernel_weight_s(kernel, xorder=order, yorder=0)
        elif kernel_type == "dy":
            return kernel_weight_s(kernel, xorder=0, yorder=order)
        else:
            raise ValueError(f"kernel_type = {kernel_type} not supported")
    else:
        raise NotImplementedError("The shape of the kernel is neither cross or square")


default_interp_funcs = [kernel_weight_x(a_kernel) for a_kernel in DEFAULT_KERNELS]


def find_which_points_for_each_kernel(masked, inheritance="default"):
    """Find which kernel to use at each point.

    masked is going to be a n*m array,
    where n is the number of points of interest.
    m is the size of the largest kernel.

    inheritance defines the shape of smaller kernels.
    say
    inheritance = [
    [0,1,2,3,4],
    [0,1],
    [0]
    ]
    it means that the largest kernel have all 5 nodes
    the second kernel only contain the first and second node,
    and the last one only have the nearest neighbor.

    if a row of matrix looks like [1,1,1,1,1],
    the index of the row will be in the first element(list) of the return
    variable.

    if a row of matrix looks like [1,1,1,1,0],
    although it fits both 2nd and 3rd kernel, 2nd has priority,
    so the index will be in the 2nd element of the return pk.

    if a row looks like [0,0,0,0,0],
    none of the kernel can fit it, so the index will not be in the
    return
    """
    if inheritance == "default":
        inheritance = DEFAULT_INHERITANCE
    already_wet = []
    for i, doll in enumerate(inheritance):
        wet_1d = masked[:, np.array(doll)].all(axis=1)
        already_wet.append(np.where(wet_1d)[0])
    point_for_each_kernel = [list(already_wet[0])]
    for i in range(1, len(inheritance)):
        point_for_each_kernel.append(
            list(np.setdiff1d(already_wet[i], already_wet[i - 1]))
        )
    return point_for_each_kernel


def get_weight_cascade(
    rx,
    ry,
    pk,
    kernel_large=DEFAULT_KERNEL,
    inheritance=DEFAULT_INHERITANCE,
    funcs=default_interp_funcs,
):
    """Compute the weight.

    apply the corresponding functions that was figured out in
    find_which_points_for_each_kernel

    Parameters
    ----------
    rx,ry: numpy.ndarray
        1D array with length N of non-dimensional relative horizontal
        position to the nearest node
    kernel_large: numpy.ndarray
        A numpy kernel of shape (M,2) that contains all the kernels needed.
    inheritance: list of list(s)
        The inheritance sequence when some of the node is masked.
    funcs: list of compileable functions
        The weight function of each kernel in the inheritance sequence.

    Returns
    -------
    + weight: numpy.ndarray
        The horizontal weight of interpolation/derivative for the points
        with shape (N,M)

    """
    weight = np.zeros((len(rx), len(kernel_large)))
    weight[:, 0] = np.nan
    for i in range(len(pk)):
        if len(pk[i]) == 0:
            continue
        sub_rx = rx[pk[i]]
        sub_ry = ry[pk[i]]
        #     slim_weight = interp_func[i](sub_rx,sub_ry)
        sub_weight = np.zeros((len(pk[i]), len(kernel_large)))
        sub_weight[:, np.array(inheritance[i])] = funcs[i](sub_rx, sub_ry)
        weight[pk[i]] = sub_weight
    return weight


def find_pk_4d(masked, inheritance=DEFAULT_INHERITANCE):
    """Find the masking condition for 4D space time.

    See find_which_points_for_each_kernel
    """
    maskedT = masked.T
    ind_shape = maskedT.shape
    pk4d = []
    for i in range(ind_shape[0]):
        z = []
        for j in range(ind_shape[1]):
            z.append(find_which_points_for_each_kernel(maskedT[i, j].T, inheritance))
        pk4d.append(z)
    return pk4d


def kash(kernel):  # hash kernel
    """Hash a horizontal kernel.

    Return the hash value.

    Parameters
    ----------
    + kernel: numpy.ndarray
        A horizontal kernel
    """
    return hash(tuple((i, j) for (i, j) in kernel))


@cache
def _get_func_from_hashable(
    kernel_tuple, kernel_shape, hkernel="interp", h_order=0, **kwargs
):
    kernel = np.array(kernel_tuple).reshape(kernel_shape)
    return kernel_weight(kernel, kernel_type=hkernel, order=h_order)


def get_func(kernel, **kwargs):
    """Return functions that compute weights.

    Similar to the kernel_weight function,
    the only difference is that this function can
    read existing functions that is cached.
    See _get_func_from_hashable

    See Also
    --------
    kernel_weight: the un-hashed version of this function.
    """
    return _get_func_from_hashable(tuple(kernel.ravel()), kernel.shape, **kwargs)


def auto_inheritance(kernel, hkernel="interp"):
    """Find a natural inheritance pattern given one horizontal kernel."""
    if hkernel == "interp":
        doll = [list(range(len(kernel)))]
    elif hkernel == "dx":
        doll = [[i for i in range(len(kernel)) if kernel[i][1] == 0]]
    elif hkernel == "dy":
        doll = [[i for i in range(len(kernel)) if kernel[i][0] == 0]]
    doll[0] = sorted(
        doll[0], key=lambda i: max(abs(kernel[i] + np.array([0.01, 0.00618])))
    )
    last = doll[-1]
    lask = np.array([kernel[i] for i in last])
    dist = round(max(np.max(abs(lask), axis=1)))
    for radius in range(dist - 1, -1, -1):
        new = [i for i in last if max(abs(kernel[i])) <= radius]
        if new != last:
            last = new
            doll.append(last)
    return doll


class KnW:
    """Kernel object.

    A class that describes anything about the
    interpolation/derivative kernel to be used.

    Parameters
    ----------
    kernel: numpy.ndarray
        The largest horizontal kernel to be used
    inheritance: list
        The inheritance sequence of the kernels
    hkernel: str
        What to do in the horizontal direction
        'interp', 'dx', or 'dy'?
    tkernel: str
        What kind of operation to do in the temporal dimension:
        'linear', 'nearest' interpolation, or 'dt'
    vkernel: str
        What kind of operation to do in the vertical:
        'linear', 'nearest' interpolation, or 'dz'
    h_order: int
        How many derivative to take in the horizontal direction.
        Zero for pure interpolation
    ignore_mask: bool
        Whether to diregard the masking of the dataset.
        You can select True if there is no
        inheritance, or if performance is a big concern.
    """

    def __init__(
        self,
        kernel=DEFAULT_KERNEL,
        inheritance="auto",  # None, or list of lists
        hkernel="interp",  # dx,dy
        vkernel="nearest",  # linear,dz
        tkernel="nearest",  # linear,dt
        h_order=0,  # depend on hkernel type
        ignore_mask=False,
    ):
        kernel_sort = np.abs(kernel + np.array([0.01, 0.00618])).sum(axis=1).argsort()
        # Avoid points having same distance
        kernel_sort_inv = kernel_sort.argsort()

        if (inheritance is not None) and (ignore_mask):
            logging.info(
                "Overwriting the inheritance object to None,"
                " because we ignore masking"
            )

        if inheritance == "auto":
            inheritance = auto_inheritance(kernel, hkernel=hkernel)
        elif inheritance is None:  # does not apply cascade
            inheritance = [list(range(len(kernel)))]
        elif isinstance(inheritance, list):
            pass
        else:
            raise ValueError("Unknown type of inherirance")

        self.kernel = kernel[kernel_sort]
        self.inheritance = [
            sorted([kernel_sort_inv[i] for i in heir]) for heir in inheritance
        ]
        self.hkernel = hkernel
        self.vkernel = vkernel
        self.tkernel = tkernel
        self.h_order = h_order
        self.ignore_mask = ignore_mask

        self.kernels = [
            np.array([self.kernel[i] for i in doll]) for doll in self.inheritance
        ]
        self.hfuncs = [
            get_func(kernel=a_kernel, hkernel=self.hkernel, h_order=self.h_order)
            for a_kernel in self.kernels
        ]

    def same_hsize(self, other):
        """Return True if 2 KnW object has the same horizontal size."""
        type_same = isinstance(other, type(self))
        if not type_same:
            raise TypeError("the argument is not a KnW object")
        try:
            return np.allclose(self.kernel, other.kernel)
        except (ValueError, AttributeError):
            return False

    def same_size(self, other):
        """Return True if 2 KnW object has the same 4D size."""
        only_size = {"dz": 2, "linear": 2, "dt": 2, "nearest": 1}
        hsize_same = self.same_hsize(other)
        vsize_same = only_size[self.vkernel] == only_size[other.vkernel]
        tsize_same = only_size[self.tkernel] == only_size[other.tkernel]
        return hsize_same and vsize_same and tsize_same

    def __eq__(self, other):
        type_same = isinstance(other, type(self))
        if not type_same:
            return False
        shpe_same = self.same_hsize(other) and self.inheritance == other.inheritance
        diff_same = (
            (self.hkernel == other.hkernel)
            and (self.vkernel == other.vkernel)
            and (self.tkernel == other.tkernel)
        )
        return type_same and shpe_same and diff_same

    def __hash__(self):
        return hash(
            (
                kash(self.kernel),
                tuple(tuple(i for i in heir) for heir in self.inheritance),
                self.ignore_mask,
                self.h_order,
                self.hkernel,
                self.vkernel,
                self.tkernel,
            )
        )

    def size_hash(self):
        """Produce a hash value based on the 4D size of the KnW object."""
        only_size = {"dz": 2, "linear": 2, "dt": 2, "nearest": 1}
        return hash(
            (kash(self.kernel), only_size[self.vkernel], only_size[self.tkernel])
        )

    def get_weight(
        self,
        rx,
        ry,
        rz=0,
        rt=0,
        pk4d=None,  # All using the largest
        bottom_scheme="no flux",  # None
    ):
        """Return the weight of values given particle rel-coords.

        Parameters
        ----------
        rx,ry,rz,rt: numpy.ndarray
            1D array of non-dimensional particle positions
        pk4d: list
            A mapping on which points should use which kernel.
        bottom_scheme: str
            Whether to assume there is a ghost point with same value at
            the bottom boundary.
            Choose None for vertical flux, 'no flux' for most other cases.

        Returns
        -------
        weight: numpy.ndarray
            The weight of interpolation/derivative for the points
            with shape (N,M),
            M is the num of node in the largest kernel.
        """
        if self.vkernel in ["linear", "dz"]:
            nz = 2
        else:
            nz = 1
        if self.tkernel in ["linear", "dt"]:
            nt = 2
        else:
            nt = 1

        weight = np.zeros((len(rx), len(self.kernel), nz, nt))
        if isinstance(rz, (int, float)) and self.vkernel != "nearest":
            rz = np.array([rz for i in range(len(rx))])
        if isinstance(rt, (int, float)) and self.tkernel != "nearest":
            rt = np.array([rt for i in range(len(rx))])

        if self.tkernel == "linear":
            rpt = copy.deepcopy(rt)
            tweight = [
                (1 - rpt).reshape((len(rpt), 1, 1)),
                rpt.reshape((len(rpt), 1, 1)),
            ]
        elif self.tkernel == "dt":
            tweight = [-1, 1]
        elif self.tkernel == "nearest":
            tweight = [1]

        if self.vkernel == "linear":
            rpz = copy.deepcopy(rz)
            zweight = [(1 - rpz).reshape((len(rpz), 1)), rpz.reshape((len(rpz), 1))]
        elif self.vkernel == "dz":
            zweight = [-1, 1]
        elif self.vkernel == "nearest":
            zweight = [1]

        if pk4d is None:
            # pk4d = [
            #    [
            #        [list(range(len(rx)))]
            # # all points are in the same catagory
            #    for i in range(nz)]# Every layer is the same
            # for j in range(nt)]#
            for jt in range(nt):
                for jz in range(nz):
                    weight[:, self.inheritance[0], jz, jt] = self.hfuncs[0](rx, ry)
        else:
            if nt != len(pk4d) or nz != len(pk4d[0]):
                raise ValueError("The kernel and the input pk4d does not match")

            for jt in range(nt):
                for jz in range(nz):
                    weight[:, :, jz, jt] = get_weight_cascade(
                        rx,
                        ry,
                        pk4d[jt][jz],
                        kernel_large=self.kernel,
                        inheritance=self.inheritance,
                        funcs=self.hfuncs,
                    )
        for jt in range(nt):
            if (self.vkernel == "linear") and (bottom_scheme == "no flux"):
                # whereever the bottom layer is masked,
                # replace it with a ghost point above it
                secondlayermasked = np.isnan(weight[:, :, 0, jt]).any(axis=1)
                # setting the value at this level zero
                weight[secondlayermasked, :, 0, jt] = 0
                shouldbemasked = np.logical_and(secondlayermasked, rz < 1 / 2)
                weight[shouldbemasked, :, 1, jt] = 0
                # setting the vertical weight of the above value to 1
                zweight[1][secondlayermasked] = 1
            for jz in range(nz):
                weight[:, :, jz, jt] *= zweight[jz]
            weight[:, :, :, jt] *= tweight[jt]

        return weight
