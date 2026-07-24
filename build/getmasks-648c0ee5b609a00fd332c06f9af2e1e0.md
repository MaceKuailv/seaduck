# seaduck.get_masks

## abandon_stuck

```python
abandon_stuck(p)
```

Abandon those stucked in mud.

---

## get_mask_arrays

```python
get_mask_arrays(od)
```

Mask all staggered valocity points.

A wrapper around mask_u_node, mask_v_node, mask_w_node.
If there is no maskC in the dataset, just return nothing is masked.

**Parameters**

- **od** (*OceData object*) -- The dataset to compute masks on.
- **tp** (*Topology object*) -- The Topology of the datset

**Returns**

- **maskC,maskU,maskV,maskW** (*numpy.ndarray*) -- masks at center points, U-walls, V-walls, W-walls respectively.

---

## get_masked

```python
get_masked(od, ind, cuvwg='C')
```

Return whether points are masked.

Return whether the indexes of intersts are masked or not.

**Parameters**

- **od** (*OceData object*) -- Dataset to find mask values from.
- **ind** (*tuple of numpy.ndarray*) -- Indexes of grid points.
- **cuvwg** (*str*) -- Whether the indexes is for points at center points or on the walls. Options are: ['C','U','V','Wvel'].

---

## mask_u_node

```python
mask_u_node(maskC, tp)
```

Mask out U-points.

for MITgcm indexing, U is defined on the left of the cell,
When the C grid is dry, U are either:
a. dry;
b. on the interface, where the cell to the left is wet.
if b is the case, we need to unmask the udata, because it makes some physical sense.

**Parameters**

- **maskC** (*numpy.ndarray*) -- numpy array with the same shape as the model spacial coordinates. 1 for wet cells (center points), 0 for dry ones.
- **tp** (*Topology object*) -- The Topology object for the dataset.

**Returns**

- **maskU** (*numpy.ndarray*) -- numpy array with the same shape as the model spacial coordinates. 1 for wet U-walls (including interface between wet and dry), 0 for dry ones.

---

## mask_v_node

```python
mask_v_node(maskC, tp)
```

Mask out v-points.

for MITgcm indexing, V is defined on the "south" side of the cell,
When the C grid is dry, V are either:
a. dry;
b. on the interface, where the cell to the downside is wet.
if b is the case, we need to unmask the vdata.

**Parameters**

- **maskC** (*numpy.ndarray*) -- numpy array with the same shape as the model spacial coordinates. 1 for wet cells (center points), 0 for dry ones.
- **tp** (*Topology object*) -- The Topology object for the dataset.

**Returns**

- **+ maskV** (*numpy.ndarray*) -- numpy array with the same shape as the model spacial coordinates. 1 for wet W-walls (including interface between wet and dry), 0 for dry ones.

---

## mask_w_node

```python
mask_w_node(maskC, tp=None)
```

Mask out W-points.

for MITgcm indexing, W is defined on the top of the cell,
When the C grid is dry, W are either:
a. dry;
b. on the interface, where the cell above is wet.
if b is the case, we need to unmask the wdata.

**Parameters**

- **maskC** (*numpy.ndarray*) -- numpy array with the same shape as the model spacial coordinates. 1 for wet cells (center points), 0 for dry ones.
- **tp** (*Topology object*) -- The Topology object for the dataset.

**Returns**

- **+ maskWvel** (*numpy.ndarray*) -- numpy array with the same shape as the model spacial coordinates. 1 for wet W-walls (including interface between wet and dry), 0 for dry ones.

---

## which_not_stuck

```python
which_not_stuck(p)
```

Investigate which points are in land mask.
