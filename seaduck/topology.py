import copy
import logging

import numpy as np

from seaduck.runtime_conf import compileable

legal_tends = [0, 1, 2, 3]  # up,down,left,right #list(llc_face_connect.columns)

llc_face_connect = np.array(
    [
        [1, 42, 12, 3],
        [2, 0, 11, 4],
        [6, 1, 10, 5],
        [4, 42, 0, 9],
        [5, 3, 1, 8],
        [6, 4, 2, 7],
        [10, 5, 2, 7],
        [10, 5, 6, 8],
        [11, 4, 7, 9],
        [12, 3, 8, 42],
        [2, 7, 6, 11],
        [1, 8, 10, 12],
        [0, 9, 11, 42],
    ]
)

directions = np.array([np.pi / 2, -np.pi / 2, np.pi, 0])


@compileable
def llc_mutual_direction(face, nface, transitive=False):
    """Find the relative orientation of two faces.

    The compileable version of mutual direction for llc grid.
    See Topology.mutual direction for more detail.
    """
    edge_n = np.where(llc_face_connect[face] == nface)
    nedge_n = np.where(llc_face_connect[nface] == face)
    if len(edge_n[0]) == 0:
        found = False
    else:
        found = edge_n[0][0] in [0, 1, 2, 3] and nedge_n[0][0] in [0, 1, 2, 3]
    if found:
        return edge_n[0][0], nedge_n[0][0]
    elif transitive:
        common = -1
        for i in llc_face_connect[face]:
            if i in llc_face_connect[nface]:
                common = i
                break
        if common < 0:
            raise ValueError(
                "The two faces does not share common face, transitive did not help"
            )
        else:
            edge_1 = np.where(llc_face_connect[face] == common)[0][0]
            nedge_1 = np.where(llc_face_connect[common] == face)[0][0]
            edge_2 = np.where(llc_face_connect[common] == nface)[0][0]
            nedge_2 = np.where(llc_face_connect[nface] == common)[0][0]
            if (edge_1 in [0, 1] and nedge_1 in [0, 1]) or (
                edge_1 in [2, 3] and nedge_1 in [2, 3]
            ):
                return edge_2, nedge_2
            elif (edge_2 in [0, 1] and nedge_2 in [0, 1]) or (
                edge_2 in [2, 3] and nedge_2 in [2, 3]
            ):
                return edge_1, nedge_1
            else:
                raise NotImplementedError(
                    "the common face is not parallel to either of the face"
                )
    else:
        raise ValueError("The two faces are not connected (transitive = False)")


@compileable
def llc_get_the_other_edge(face, edge):
    """See what is adjacent to the face by this edge.

    The compileable version of get_the_other_edge for llc grid.
    See Topology.get_the_other_edge for more detail.
    """
    face_connect = llc_face_connect
    nface = face_connect[face, edge]
    if nface == 42:
        raise IndexError(
            "Reaching the edge where the face is not connected to any other face"
        )
    nedge_n = np.where(face_connect[nface] == face)
    return nface, nedge_n[0][0]


@compileable
def box_ind_tend(ind, tend, iymax, ixmax):
    """Move an index in a direction.

    The compileable version of ind_tend for regional (box) grid.
    See Topology.ind_tend for more detail.
    """
    iy, ix = ind
    if tend == 0:
        iy += 1
    elif tend == 1:
        iy -= 1
    elif tend == 2:
        ix -= 1
    elif tend == 3:
        ix += 1
    # it would be better to raise an error here.
    if (iy > iymax) or (iy < 0):
        return (-1, -1)
    if (ix > ixmax) or (ix < 0):
        return (-1, -1)
    return (iy, ix)


@compileable
def x_per_ind_tend(ind, tend, iymax, ixmax):
    """Move an index in a direction.

    The compileable version of ind_tend for zonally periodic (x-per) grid.
    See Topology.ind_tend for more detail.
    """
    iy, ix = ind
    if tend == 0:
        iy += 1
    elif tend == 1:
        iy -= 1
    elif tend == 2:
        ix -= 1
    elif tend == 3:
        ix += 1
    if (iy > iymax) or (iy < 0):
        return (-1, -1)
    if ix > ixmax:
        return (iy, ix - ixmax)
    if ix < 0:
        return (iy, ixmax + ix + 1)
    return (iy, ix)


@compileable
def llc_ind_tend(ind, tendency, iymax, ixmax):
    """Move an index in a direction.

    The compileable version of ind_tend for llc grid.
    See Topology.ind_tend for more detail.
    """
    #     iymax = 89
    #     ixmax = 89
    face, iy, ix = ind
    if tendency == 3:
        if ix != ixmax:
            ix += 1
        else:
            nface, nedge = llc_get_the_other_edge(face, 3)
            if nedge == 1:
                face, iy, ix = [nface, 0, ixmax - iy]
            elif nedge == 0:
                face, iy, ix = [nface, iymax, iy]
            elif nedge == 2:
                face, iy, ix = [nface, iy, 0]
            elif nedge == 3:
                face, iy, ix = [nface, iymax - iy, ixmax]
    if tendency == 2:
        if ix != 0:
            ix -= 1
        else:
            nface, nedge = llc_get_the_other_edge(face, 2)
            if nedge == 1:
                face, iy, ix = [nface, 0, iy]
            elif nedge == 0:
                face, iy, ix = [nface, iymax, ixmax - iy]
            elif nedge == 2:
                face, iy, ix = [nface, iymax - iy, 0]
            elif nedge == 3:
                face, iy, ix = [nface, iy, ixmax]
    if tendency == 0:
        if iy != iymax:
            iy += 1
        else:
            nface, nedge = llc_get_the_other_edge(face, 0)
            if nedge == 1:
                face, iy, ix = [nface, 0, ix]
            elif nedge == 0:
                face, iy, ix = [nface, iymax, ixmax - ix]
            elif nedge == 2:
                face, iy, ix = [nface, iymax - ix, 0]
            elif nedge == 3:
                face, iy, ix = [nface, ix, ixmax]
    if tendency == 1:
        if iy != 0:
            iy -= 1
        else:
            nface, nedge = llc_get_the_other_edge(face, 1)
            if nedge == 1:
                face, iy, ix = [nface, 0, ixmax - ix]
            elif nedge == 0:
                face, iy, ix = [nface, iymax, ix]
            elif nedge == 2:
                face, iy, ix = [nface, ix, 0]
            elif nedge == 3:
                face, iy, ix = [nface, iymax - ix, ixmax]
    return (face, iy, ix)


@compileable
def llc_get_uv_mask_from_face(faces):
    """Get the masking of UV points.

    The compileable version of get_uv_mask_from_face for llc grid.
    See Topology.get_uv_mask_from_face for more detail.
    """
    # we are considering a row from the fatten_face
    # faces is essentially which face each node of the kernel is on.
    n = len(faces)  # it should have been m to be more consistent with other code
    UfromUvel = np.ones(n)
    UfromVvel = np.zeros(n)
    VfromUvel = np.zeros(n)
    VfromVvel = np.ones(n)
    # if all of the node are on the same face, we don't have to convert anything
    if np.abs(np.ones(n) * faces[0] - faces).max() < 1e-5:
        return UfromUvel, UfromVvel, VfromUvel, VfromVvel
    else:
        for i in range(1, n):
            if faces[i] == faces[0]:
                continue
            # if the face is not the same, we need to do something
            else:
                # get how much the new face is rotated from the old face
                edge, nedge = llc_mutual_direction(faces[0], faces[i], transitive=True)
                rot = np.pi - directions[edge] + directions[nedge]
                # you can think of this as a rotation matrix
                UfromUvel[i] = np.cos(rot)
                UfromVvel[i] = np.sin(rot)
                VfromUvel[i] = -np.sin(rot)
                VfromVvel[i] = np.cos(rot)
        return UfromUvel, UfromVvel, VfromUvel, VfromVvel


class Topology:
    """Topology object.

    A light weight object that remembers the structure of the grid,
    what is connected, what is not. The core method is simply move the index to a direction.
    In the movie kill Bill, the bride sat in a car and said "wiggle your big toe".
    After the toe moved, "the hard part is over". You get the idea.

    Parameters
    ----------
    od: xarray.Dataset
        The dataset to record topological info from.
    typ: None, or str
        Type of the grid.
        Currently we support
        'box' for regional dataset,
        'x-periodic' for zonally periodic ones,
        'llc' for lat-lon-cap dataset.
        We recommend that users put None here,
        so that the type is figured out automatically.
    """

    def __init__(self, od, typ=None):
        try:
            h_shape = od["XC"].shape
        except KeyError:
            try:
                h_shape = (int(od["lat"].shape[0]), int(od["lon"].shape[0]))
            except KeyError as exc:
                raise KeyError(
                    "Either XC or lat/lon is needed to create the Topology object"
                ) from exc
        self.h_shape = h_shape
        if "XG" in od.variables:
            self.g_shape = od["XG"].shape
        else:
            self.g_shape = None
        try:
            self.itmax = len(od["time"]) - 1
        except (KeyError, TypeError):
            self.itmax = 0
        try:
            self.izmax = len(od["Z"]) - 1
        except (KeyError, TypeError):
            self.izmax = 0

        if typ:
            self.typ = typ
        elif typ is None:
            if len(h_shape) == 3:
                self.num_face, self.iymax, self.ixmax = h_shape
                self.iymax -= 1
                self.ixmax -= 1
                if self.num_face == 13:
                    self.typ = "LLC"
                    # we can potentially generate the face connection in runtime
                    # say, put the csv file on cloud
                else:
                    raise NotImplementedError(
                        "Other complex topology is not implemented yet."
                        "If you want to work with such data, please contact the authors."
                    )
            elif len(h_shape) == 2:
                self.iymax, self.ixmax = h_shape
                self.iymax -= 1
                self.ixmax -= 1
                try:
                    lon_right = float(od["XC"][0, self.ixmax])
                    lon_left = float(od["XC"][0, 0])
                except KeyError:
                    lon_right = float(od["lon"][self.ixmax])
                    lon_left = float(od["lon"][0])
                left_to_right = (lon_right - lon_left) % 360
                right_to_left = (lon_left - lon_right) % 360
                if left_to_right > 50 * right_to_left:
                    self.typ = "x_periodic"
                else:
                    self.typ = "box"

    def get_the_other_edge(self, face, edge):
        """See what is adjacent to the face by this edge.

        The (edge) side of the (face) is connected to
        the (nedge) side of the (nface).
        0,1,2,3 stands for up, down, left, right.

        Parameters
        ----------
        face: int
            The face of interst
        edge: 0,1,2,3
            which direction of the face we are looking for

        Returns
        -------
        nface: int
            The face adjacent to face in the edge direction.
        nedge: 0,1,2,3
            The face is connected to nface in which direction.
        """
        if self.typ == "LLC":
            return llc_get_the_other_edge(face, edge)
        elif self.typ in ["x_periodic", "box"]:
            raise ValueError(
                "It makes no sense to tinker with face_connection when there is only one face"
            )
        else:
            raise NotImplementedError

    def mutual_direction(self, face, nface, **kwarg):
        """Find the relative orientation of two faces.

        0,1,2,3 stands for up, down, left, right
        given 2 faces, the returns are
        1. the 2nd face is to which direction of the 1st face
        2. the 1st face is to which direction of the 2nd face.
        """
        if self.typ == "LLC":
            return llc_mutual_direction(face, nface, **kwarg)
        elif self.typ in ["x_periodic", "box"]:
            raise ValueError(
                "It makes no sense to tinker with face_connection when there is only one face"
            )
        else:
            raise NotImplementedError

    def ind_tend(self, ind, tend, cuvwg="C", **kwarg):
        """Move an index in a direction.

        ind is a tuple that is face,iy,ix,
        tendency again is up, down, left, right represented by 0,1,2,3
        return the next cell.

        Parameters
        ----------
        ind: tuple
            The index to find the neighbor of
        tend: int
            Which direction to move from the original index.
        cuvwg: str, default "C"
            Whether we are moving from C grid, U grid, V grid, or G grid.
        kwarg:dict, optional
            Keyword argument that currently only apply for the llc case.
        """
        if -1 in ind:
            # meaning invalid point
            return tuple(-1 for i in ind)
        if self.typ == "LLC":
            if cuvwg == "C":
                to_return = llc_ind_tend(ind, tend, self.iymax, self.ixmax, **kwarg)
            elif cuvwg == "U":
                _, to_return = self._ind_tend_U(ind, tend)
            elif cuvwg == "V":
                _, to_return = self._ind_tend_V(ind, tend)
            elif cuvwg == "G":
                to_return = self._ind_tend_G(ind, tend)
            else:
                raise ValueError("The type of grid point should be among C,U,V,G")
        elif self.typ == "x_periodic":
            to_return = x_per_ind_tend(ind, tend, self.iymax, self.ixmax, **kwarg)
        elif self.typ == "box":
            to_return = box_ind_tend(ind, tend, self.iymax, self.ixmax, **kwarg)
        else:
            raise NotImplementedError
        return tuple(int(i) for i in to_return)

    def ind_moves(self, ind, moves, **kwarg):
        """Move an index in a serie of directions.

        moves being a list of directions (0,1,2,3),
        ind being the starting index,
        return the index after moving in the directions in the list.

        Parameters
        ----------
        ind: tuple
            Index of the starting position
        moves: iterable
            A sequence of steps to "walk" from the original position.
        kwarg: dict, optional
            Keyword arguments that pass into ind_tend.
        """
        if self.check_illegal(ind):
            return tuple(-1 for i in ind)  # the origin is invalid
        if not set(moves).issubset({0, 1, 2, 3}):
            raise ValueError("Illegal move. Must be 0,1,2,3")
        if self.typ in ["LLC", "cubed_sphere"]:
            face, iy, ix = ind
            for k in range(len(moves)):
                move = moves[k]
                ind = self.ind_tend(ind, move, **kwarg)
                if ind[0] != face:  # if the face has changed
                    # there are times where the the kernel lies between
                    # 2 faces that define 'left' differently. That's why
                    # when that happens we need to correct the direction
                    # you want to move the indexes.
                    edge, nedge = self.mutual_direction(face, ind[0], transitive=True)
                    rot = (np.pi - directions[edge] + directions[nedge]) % (np.pi * 2)
                    if np.isclose(rot, 0):
                        pass
                    elif np.isclose(rot, np.pi / 2):
                        moves[k + 1 :] = [[2, 3, 1, 0][move] for move in moves[k + 1 :]]
                    elif np.isclose(rot, 3 * np.pi / 2):
                        moves[k + 1 :] = [[3, 2, 0, 1][move] for move in moves[k + 1 :]]
                    face = ind[0]
                    # if the old face is on the left of the new face,
                    # the particle should be heading right
        elif self.typ in ["x_periodic", "box"]:
            for move in moves:
                ind = self.ind_tend(ind, move)
        return tuple(int(i) for i in ind)

    def check_illegal(self, ind, cuvwg="C"):
        """Check if the index is legal.

        A vectorized check to see whether the index is legal,
        index can be a tuple of numpy ndarrays.
        no negative index is permitted for sanity reason.

        Parameters
        ----------
        ind: tuple
            Each element of the tuple is iterable of one dimension of the indexes.
        cuvwg: 'C' or 'G'
            Whether use the center grid or the corner grid.
        """
        if cuvwg == "C":
            the_shape = self.h_shape
        else:
            the_shape = self.g_shape

        if isinstance(ind[0], int):  # for single item
            result = False
            for i, z in enumerate(ind):
                max_pos = the_shape[i]
                if not 0 <= z <= max_pos - 1:
                    result = True
            return result
        else:  # for numpy ndarray
            result = np.zeros_like(ind[0])
            result = False  # make it cleaner
            for i, z in enumerate(ind):
                max_pos = the_shape[i]
                result = np.logical_or(
                    np.logical_or((0 > z), (z > max_pos - 1)), result
                )
            return result

    def ind_tend_vec(self, inds, tend, **kwarg):
        """Move many indices in a list of directions.

        Vectorized version for ind_tend method.

        Parameters
        ----------
        inds: tuple of numpy.array or numpy.array
            Each element of the tuple is iterable of one dimension of the indexes.
        tend: iterable
            Which direction should each index take.
        kwarg: dict,optional
            Keyword argument that feeds into ind_tend.
        """
        inds = np.array(inds)
        old_inds = copy.deepcopy(inds)
        move_dic = {
            0: np.array([1, 0]),  # delta_y,delta_x
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
        }
        naive_move = np.array([move_dic[i] for i in tend], dtype=int).T
        inds[-2:] += naive_move
        cuvwg = kwarg.get("cuvwg", "C")
        illegal = self.check_illegal(inds, cuvwg=cuvwg)
        redo = np.array(np.where(illegal)).T
        particle_on_edge = False
        for num, loc in enumerate(redo):
            j = loc[0]
            ind = tuple(old_inds[:, j])
            try:
                n_ind = self.ind_tend(ind, int(tend[j]), **kwarg)
            except IndexError:
                particle_on_edge = True
                n_ind = ind
            inds[:, j] = np.array(n_ind).ravel()
        if particle_on_edge:
            logging.warning("Some points are on the edge")
        for i in range(len(inds)):
            inds[i] = inds[i].astype(int)
        return inds

    def _find_wall_between(self, ind1, ind2):
        """Return the wall between two adjacent cells.

        if the input is not adjacent, error may not be raised
        This scheme is only valid if there is face in the dimensions.
        """
        (fc1, _, _) = ind1
        (fc2, _, _) = ind2
        Non_normal_connection = ValueError(
            f"The two face connecting the indexes {ind1},{ind2}"
            " are not connected in a normal way"
        )
        if fc1 == fc2:
            to_return = tuple(
                np.ceil((np.array(ind1) + np.array(ind2)) / 2).astype(int)
            )
            other = ind1 if ind2 == to_return else ind2
            (_, iyr, ixr) = to_return
            (_, iyo, ixo) = other
            if ixr > ixo:
                return "U", to_return
            elif iyr > iyo:
                return "V", to_return
            else:
                raise IndexError("there is no wall between a cell and itself")
                # This error can be raised when there is three instead of four points in a corner
        else:
            d1to2, d2to1 = self.mutual_direction(fc1, fc2)
            if d1to2 in [0, 3]:
                to_return = ind2
                if d2to1 == 1:
                    return "V", to_return
                elif d2to1 == 2:
                    return "U", to_return
                else:
                    raise Non_normal_connection
            elif d2to1 in [0, 3]:
                to_return = ind1
                if d1to2 == 1:
                    return "V", to_return
                elif d1to2 == 2:
                    return "U", to_return
                else:
                    raise Non_normal_connection
            else:
                raise Non_normal_connection

    def _ind_tend_U(self, ind, tend):
        """Move an U-index in a direction.

        A subset of ind_tend that deal with special issues
        that comes from staggered grid. Read ind_tend for more info.
        """
        # TODO: implement different grid offset case
        if tend in [2, 3]:
            return "U", self.ind_tend(ind, tend)
        else:
            first = self.ind_tend(ind, tend)
            if first[0] == ind[0]:
                return "U", first
            else:
                alter = self.ind_moves(ind, [2, tend])
                # TODO: Add the case of alter == first for three-way join of faces.Low priority
                return self._find_wall_between(first, alter)

    def _ind_tend_V(self, ind, tend):
        """Move a V-index in a direction.

        A subset of ind_tend that deal with special issues
        that comes from staggered grid. Read ind_tend for more info.
        """
        # TODO: implement different grid offset case
        if tend in [0, 1]:
            return "V", self.ind_tend(ind, tend)
        else:
            first = self.ind_tend(ind, tend)
            if first[0] == ind[0]:
                return "V", first
            else:
                alter = self.ind_moves(ind, [1, tend])
                return self._find_wall_between(first, alter)

    def _ind_tend_G(self, ind, tend):
        """Move a G-index(corner point) in a direction.

        If there is a index for the corner point at all,
        it needs to be connected to the two edges with the same index.
        Therefore, this will always work, regardless of the grid.
        """
        if tend in [0, 1]:
            _, nind = self._ind_tend_U(ind, tend)
            return nind
        elif tend in [2, 3]:
            _, nind = self._ind_tend_V(ind, tend)
            return nind
        else:
            raise ValueError(f"tend {tend} not supported")

    def get_uv_mask_from_face(self, faces):
        """Get the masking of UV points.

        The background is as following:
        For a dataset with face connection,
        when one is finding the neighboring cells for vector interpolation,
        the fact that faces can be rotated against each other can create
        local inconsistency in vector definition near face connections.
        This method corrects that.

        Parameters
        ----------
        faces: iterable
            1D iterable of faces, the first one is assumed to be the reference.
        """
        if self.typ == "LLC":
            return llc_get_uv_mask_from_face(faces)
        elif self.typ in ["x_periodic", "box"]:
            raise ValueError(
                "It makes no sense to tinker with face_connection when there is only one face"
            )
        else:
            raise NotImplementedError

    def four_matrix_for_uv(self, fface):
        """Expand get_uv_mask_from_face to 2D array of faces."""
        # apply get_uv_mask for the n*m matrix
        UfromUvel, UfromVvel, VfromUvel, VfromVvel = (
            np.zeros(fface.shape) for i in range(4)
        )
        for i in range(fface.shape[0]):
            (
                UfromUvel[i],
                UfromVvel[i],
                VfromUvel[i],
                VfromVvel[i],
            ) = self.get_uv_mask_from_face(fface[i])
        return UfromUvel, UfromVvel, VfromUvel, VfromVvel
