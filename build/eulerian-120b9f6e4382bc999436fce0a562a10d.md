# seaduck.eulerian

## Position

```python
class Position(*arg, **kwarg)
```

The Position object that performs the interpolation.

Create a empty one by default.
To actually do interpolation, use from_latlon method to tell the ducks where they are.

### Position.fatten   *(method)*

```python
fatten(self, knw, four_d=False, required='all', ind_moves_kwarg={})
```

Fatten in all the required dimensions.

Finding the neighboring center grid points in all 4 dimensions.

**Parameters**

- **knw** (*KnW object*) -- The kernel used to find neighboring points.
- **four_d** (*Boolean, default False*) -- When we are doing nearest neighbor interpolation on some of the dimensions, with four_d = True, this will create dimensions with length 1, and will squeeze the dimension if four_d = False
- **required** (*str, iterable of str, default "all"*) -- Which dims is needed in the process
- **ind_moves_kward** (*dict, optional*) -- Key word argument to put into ind_moves method of the Topology object. Read Topology.ind_moves for more detail.

### Position.from_bool_array   *(method)*

```python
from_bool_array(self, t=None, data=None, bool_array=None, num=None, random_seed=None)
```

Update/Generate new object with random points in given grid boxes.

Use the methods from the ocedata to transform
from lat-lon-dep-time coords to rel-coords
store the output in the Position object.

**Parameters**

- **t** (*numpy.ndarray, float or None, default None*) -- 1D array of the time coords
- **data** (*OceData object*) -- The field where the Positions are defined on.
- **bool_array** (*numpy.ndarray, or xr.DataArray*) -- Points are generated where it is True. It could be an array of tracer concentration as well.
- **num** (*int*) -- Total number of particles to seed (approximately).
- **random_seed** (*int optional*) -- The random seed used for reproducible results.

### Position.from_latlon   *(method)*

```python
from_latlon(self, x=None, y=None, z=None, t=None, data=None)
```

Fill in the coord info using lat-lon-dep-time dims.

Use the methods from the ocedata to transform
from lat-lon-dep-time coords to rel-coords
store the output in the Position object.

**Parameters**

- **x,y,z,t** (*numpy.ndarray, float or None, default None*) -- 1D array of the lat-lon-dep-time coords
- **data** (*OceData object*) -- The field where the Positions are defined on.

### Position.get_f_node_weight   *(method)*

```python
get_f_node_weight(self)
```

Find weight for the corner points interpolation.

### Position.get_px_py   *(method)*

```python
get_px_py(self)
```

Get the nearest 4 corner points of the given point.

Used for oceanparcel style horizontal interpolation.

**Returns**

- **px** (*numpy.ndarray*) -- the longitude of the Position's surrounding 4 corner points.
- **py** (*numpy.ndarray*) -- the latitude of those points mentioned above.

### Position.interpolate   *(method)*

```python
interpolate(self, var_name, knw, vec_transform=True, prefetched=None, prefetch_prefix=None)
```

Do interpolation.

This is the method that does the actual interpolation/derivative.
It is a combination of the following methods:
_register_interpolation_input,
_fatten_required_index_and_register,
_transform_vector_and_register,
_read_data_and_register,
_mask_value_and_register,
_compute_weight_and_registe,.

**Parameters**

- **var_name** (*list, str, tuple*) -- The variables to interpolate. Tuples are used for horizontal vectors. Put str and list in a list if you have multiple things to interpolate. This input also defines the format of the output.
- **knw** (*KnW object, tuple, list, dict*) -- The kernel object used for the operation. Put them in the same order as var_name. Some level of automatic broadcasting is also supported.
- **vec_transform** (*Boolean*) -- Whether to project the vector field to the local zonal/meridional direction.
- **prefetched** (*numpy.ndarray, tuple, list, dict, None, default None*) -- The prefetched array for the data, this will effectively overwrite var_name. Put them in the same order as var_name. Some level of automatic broadcasting is also supported.
- **prefetch_prefix** (*tuple, list, dict, None, default None*) -- The prefix of the prefetched array. Put them in the same order as var_name. Some level of automatic broadcasting is also supported.

**Returns**

- **to_return** (*list, numpy.array, tuple*) -- The interpolation/derivative output in the same format as var_name.

### Position.subset   *(method)*

```python
subset(self, which)
```

Create a subset of the Position object.

**Parameters**

- **which** (*numpy.ndarray, optional*) -- Define which points survive the subset operation. It be an array of either boolean or int. The selection is similar to that of selecting from a 1D numpy array.

**Returns**

- **the_subset** (*Position object*) -- The selected Positions.

### Position.update_from_subset   *(method)*

```python
update_from_subset(self, sub, which)
```

Update from the original one from a subset of the Position object.

**Parameters**

- **sub** (*Position object*) -- The Position object to be updated from.
- **which** (*numpy.ndarray, optional*) -- Define which points correpond to the subset It be an array of either boolean or int. The selection is similar to that of selecting from a 1D numpy array.
