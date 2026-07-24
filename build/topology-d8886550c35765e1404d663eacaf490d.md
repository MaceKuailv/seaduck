# seaduck.topology

## Topology

```python
class Topology(od, typ=None)
```

Topology object.

A light weight object that remembers the structure of the grid,
what is connected, what is not. The core method is simply move the index to a direction.
In the movie kill Bill, the bride sat in a car and said "wiggle your big toe".
After the toe moved, "the hard part is over". You get the idea.

**Parameters**

- **od** (*xarray.Dataset*) -- The dataset to record topological info from.
- **typ** (*None, or str*) -- Type of the grid. Currently we support 'box' for regional dataset, 'x-periodic' for zonally periodic ones, 'LLC' for lat-lon-cap dataset. We recommend that users put None here, so that the type is figured out automatically.

### Topology.check_illegal   *(method)*

```python
check_illegal(self, ind, cuvwg='C')
```

Check if the index is legal.

A vectorized check to see whether the index is legal,
index can be a tuple of numpy ndarrays.
no negative index is permitted for sanity reason.

**Parameters**

- **ind** (*tuple*) -- Each element of the tuple is iterable of one dimension of the indexes.
- **cuvwg** (*'C' or 'G'*) -- Whether use the center grid or the corner grid.

### Topology.four_matrix_for_uv   *(method)*

```python
four_matrix_for_uv(self, fface)
```

Expand get_uv_mask_from_face to 2D array of faces.

### Topology.get_the_other_edge   *(method)*

```python
get_the_other_edge(self, face, edge)
```

See what is adjacent to the face by this edge.

The (edge) side of the (face) is connected to
the (new_edge) side of the (new_face).
0,1,2,3 stands for up, down, left, right.

**Parameters**

- **face** (*int*) -- The face of interst
- **edge** (*0,1,2,3*) -- which direction of the face we are looking for

**Returns**

- **new_face** (*int*) -- The face adjacent to face in the edge direction.
- **new_edge** (*0,1,2,3*) -- The face is connected to new_face in which direction.

### Topology.get_uv_mask_from_face   *(method)*

```python
get_uv_mask_from_face(self, faces)
```

Get the masking of UV points.

The background is as following:
For a dataset with face connection,
when one is finding the neighboring cells for vector interpolation,
the fact that faces can be rotated against each other can create
local inconsistency in vector definition near face connections.
This method corrects that.

**Parameters**

- **faces** (*iterable*) -- 1D iterable of faces, the first one is assumed to be the reference.

### Topology.ind_moves   *(method)*

```python
ind_moves(self, ind, moves, **kwarg)
```

Move an index in a serie of directions.

moves being a list of directions (0,1,2,3),
ind being the starting index,
return the index after moving in the directions in the list.

**Parameters**

- **ind** (*tuple*) -- Index of the starting position
- **moves** (*iterable*) -- A sequence of steps to "walk" from the original position.
- **kwarg** (*dict, optional*) -- Keyword arguments that pass into ind_tend.

### Topology.ind_tend   *(method)*

```python
ind_tend(self, ind, tend, cuvwg='C', **kwarg)
```

Move an index in a direction.

ind is a tuple that is face,iy,ix,
tendency again is up, down, left, right represented by 0,1,2,3
return the next cell.

**Parameters**

- **ind** (*tuple*) -- The index to find the neighbor of
- **tend** (*int*) -- Which direction to move from the original index.
- **cuvwg** (*str, default "C"*) -- Whether we are moving from C grid, U grid, V grid, or G grid.
- **kwarg** (*dict, optional*) -- Keyword argument that currently only apply for the llc case.

### Topology.ind_tend_vec   *(method)*

```python
ind_tend_vec(self, inds, tend, **kwarg)
```

Move many indices in a list of directions.

Vectorized version for ind_tend method.

**Parameters**

- **inds** (*tuple of numpy.array or numpy.array*) -- Each element of the tuple is iterable of one dimension of the indexes.
- **tend** (*iterable*) -- Which direction should each index take.
- **kwarg** (*dict,optional*) -- Keyword argument that feeds into ind_tend.

### Topology.mutual_direction   *(method)*

```python
mutual_direction(self, face, new_face, **kwarg)
```

Find the relative orientation of two faces.

0,1,2,3 stands for up, down, left, right
given 2 faces, the returns are
1. the 2nd face is to which direction of the 1st face
2. the 1st face is to which direction of the 2nd face.
