# seaduck.lagrangian

## Particle

```python
class Particle(uname='UVELMASS', vname='VVELMASS', wname='WVELMASS', free_surface='noflux', save_raw=False, transport=False, callback=None, max_iteration=200, **kwarg)
```

Lagrangian particle object.

The Lagrangian particle object. Simply a eulerian Position object
that know how to move itself.

**Parameters**

- **kwarg** (*dict*) -- The keyword argument that feed into Position.from_latlon method
- **uname, vname, wname** (*str*) -- The variable names for the velocity/mass-transport components. If transport is true, pass in names of the volume/mass transport across cell wall in m^3/3 else,  just pass something that is in m/s
- **free_surface** (*string*) -- Sometimes there is non-zero vertical velocity at sea surface. free_surface = "noflux" set that to zero. free_surface = "kick_back" move particles trying to cross back to the middle of the cell. There could be errors if neither is used.
- **save_raw** (*Boolean*) -- Whether to record the analytical history of all particles in an unstructured list.
- **transport** (*Boolean*) -- If transport is true, pass in names of the volume/mass transport across cell wall in m^3/3 else,  just pass velocity that is in m/s
- **callback** (*function that take Particle as input*) -- A callback function that takes Particle as input. Return boolean array that determines which Particle should still be going. Users can also define customized functions here.
- **max_iteration** (*int*) -- The number of analytical steps allowed for the to_next_stop method.

### Particle.analytical_step   *(method)*

```python
analytical_step(self, tf)
```

Integrate the particle with velocity.

The core method.
A set of particles trying to integrate for time tf
(could be negative).
at the end of the call, every particle are either:
1. ended up somewhere within the cell after time tf.
2. ended up on a cell wall before
(if tf is negative, then "after") tf.

**Parameters**

- **tf** (*float, numpy.ndarray*) -- The longest duration of the simulation for each particle.

### Particle.cross_cell_wall   *(method)*

```python
cross_cell_wall(self, tend)
```

Update properties after particles cross wall.

This function is called when particle reached the wall.
The nearest grid points change as well as the way the package
describe the location of particles. This method handles the
handover of particles between grid points.

**Parameters**

- **tend** (*numpy.ndarray of [0,1,2,3,4,5,6]*) -- Which neighboring cell to move into. 0-6 means left, right, down, up, deep, shallow, and stay in the current cell, respectively.

### Particle.deepcopy   *(method)*

```python
deepcopy(self)
```

Return a clone of the object.

### Particle.empty_lists   *(method)*

```python
empty_lists(self)
```

Empty/Create the lists.

Some times the raw-data list get too long,
It would be necessary to dump the data,
and empty the lists containing the raw data.
This method does the latter.

### Particle.fatten   *(method)*

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

### Particle.fillna   *(method)*

```python
fillna(self)
```

Fill the np.nan values to nan.

This is just to let those in rock stay in rock.

### Particle.from_bool_array   *(method)*

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

### Particle.from_latlon   *(method)*

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

### Particle.get_f_node_weight   *(method)*

```python
get_f_node_weight(self)
```

Find weight for the corner points interpolation.

### Particle.get_px_py   *(method)*

```python
get_px_py(self)
```

Get the nearest 4 corner points of the given point.

Used for oceanparcel style horizontal interpolation.

**Returns**

- **px** (*numpy.ndarray*) -- the longitude of the Position's surrounding 4 corner points.
- **py** (*numpy.ndarray*) -- the latitude of those points mentioned above.

### Particle.get_u_du   *(method)*

```python
get_u_du(self)
```

Read the velocity at particle position.

Read the velocity and velocity derivatives in all three dimensions
using the interpolate method with the default kernel.
Read eulerian.Position.interpolate for more detail.

### Particle.get_vol   *(method)*

```python
get_vol(self)
```

Read in the volume of the cell.

For particles that has transport = True,
volume of the cell is needed for the integration.
This method read the volume that is calculated at __init__.

### Particle.interpolate   *(method)*

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

### Particle.note_taking   *(method)*

```python
note_taking(self, subset_index=None, stamp=-1)
```

Record raw data into list of lists.

This method is only called in save_raw = True particles.
This method will note done the raw info of the particle
trajectories.
With those info, one could reconstruct the analytical
trajectories to arbitrary position.

**Parameters**

- **subset_index** (*iterable of int or None*) -- if not None, assume this method is called from a subset of the full particle object, and subset_index is the indices the subset occupy in the original object.

### Particle.subset   *(method)*

```python
subset(self, which)
```

Create a subset of the Position object.

**Parameters**

- **which** (*numpy.ndarray, optional*) -- Define which points survive the subset operation. It be an array of either boolean or int. The selection is similar to that of selecting from a 1D numpy array.

**Returns**

- **the_subset** (*Position object*) -- The selected Positions.

### Particle.to_list_of_time   *(method)*

```python
to_list_of_time(self, normal_stops, update_stops='default', return_in_between=True, dump_filename=False, store_kwarg={})
```

Integrate the particles to a list of time.

**Parameters**

- **normal_stops** (*iterable*) -- The time steps that user request a output
- **update_stops** (*iterable, or 'default'*) -- The time steps that uvw array changes in the model. If 'default' is set, the method is going to figure it out automatically.
- **return_in_between** (*Boolean*) -- Users can get the values of update_stops free of computational cost.We understand that user may sometimes don't want those in the output.In that case, it that case, set it to be False, and the output will all be at normal_stops.

**Returns**

- **stops** (*list*) -- The list of stops. It is the combination of normal_stops and output_stops by default. f return_in_between is set to be False, this is then the same as normal stops.
- **to_return** (*list*) -- A list deep copy of particle that inherited the interpolate method as well as velocity and coords info.

### Particle.to_next_stop   *(method)*

```python
to_next_stop(self, t_stop)
```

Integrate all particles towards time tl.

This is done by repeatedly calling analytical step.
Or at least try to do so before maximum_iteration is reached.
If the maximum time is reached,
we also force all particle's internal clock to be tl.

**Parameters**

- **t_stop** (*float*) -- The final time relative to 1970-01-01 in seconds.

### Particle.trim   *(method)*

```python
trim(self, tol=0.0)
```

Move the particles from outside the cell into the cell.

At the same time change the velocity accordingly.
In the mean time, creating some negiligible error in time.

**Parameters**

- **tol** (*float*) -- The relative tolerance when particles is significantly close to the cell.

### Particle.update_from_subset   *(method)*

```python
update_from_subset(self, sub, which)
```

Update from the original one from a subset of the Position object.

**Parameters**

- **sub** (*Position object*) -- The Position object to be updated from.
- **which** (*numpy.ndarray, optional*) -- Define which points correpond to the subset It be an array of either boolean or int. The selection is similar to that of selecting from a 1D numpy array.

### Particle.update_uvw_array   *(method)*

```python
update_uvw_array(self)
```

Update the prefetched velocity arrays.

The way to do it is slightly different for dataset with time
dimensions and those without.
