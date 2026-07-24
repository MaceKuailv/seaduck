# seaduck.utils

## NoneIn

```python
NoneIn(lst)
```

See if there is a None in the iterable object. Return a Boolean.

---

## chg_ref_lon

```python
chg_ref_lon(x, ref_lon)
```

Change the definition of 0 longitude.

Return how much east one need to go from ref_lon to x
This function aims to address
the confusion caused by the discontinuity in longitude.

---

## convert_time

```python
convert_time(time)
```

Convert time into seconds after 1970-01-01.

time needs to be a string or a np.datetime64 object.

---

## create_tree

```python
create_tree(x, y, R=6371.0, leafsize=16)
```

Create a cKD tree object.

**Parameters**

- **x,y** (*np.ndarray*) -- longitude and latitude of the grid location
- **R** (*float*) -- The radius in kilometers of the planet.
- **leafsize** (*int*) -- When to switch to brute force search.

---

## easy_3d_cube

```python
easy_3d_cube(lon, lat, dep, tim, print_total_number=False)
```

Create 4D coords for initializing Position/Particle.

---

## find_cs_sn

```python
find_cs_sn(thetaA, phiA, thetaB, phiB)
```

Find a spherical angle OAB.

theta is the angle
between the meridian crossing point A
and the geodesic connecting A and B.

this function return cos and sin of theta

---

## find_ind

```python
find_ind(array, value, peri=None, ascending=1, above=True)
```

Find the index of the nearest value to the given value.

**Parameters**

- **array** (*numpy.ndarray*) -- 1D numpy array to search index from
- **value** (*number*) -- The value to find nearest neighbor with
- **peri** (*number, optional*) -- The periodicity of the array. For example, 360 for longitude.
- **ascending** (*int, default 1*) -- Whether the array is in ascending order. 1 for ascending order, -1 for descending order.
- **above** (*boolean, default True*) -- If True, return the index of the largest item in array smaller than value. Otherwise, return the closest value.

---

## find_ind_h

```python
find_ind_h(lons, lats, tree, h_shape)
```

Use ckd tree to find the horizontal indexes,.

---

## find_px_py

```python
find_px_py(XG, YG, tp, ind, cuvwg='G')
```

Find the nearest 4 corner points.

This is used in oceanparcel interpolation scheme.

---

## find_rel

```python
find_rel(value, array, darray=None, ascending=1, above=True, peri=None, dx_right=True)
```

Find the rel-coords of the 1D coords.

The backend for all find_rel functions

**Parameters**

- **value** (*numpy.ndarray*) -- 1D array for the value to find rel-coords.
- **array** (*numpy.ndarray*) -- The array of potential reference levels.
- **darray** (*numpy.ndarray, optional*) -- The distances between reference levels.
- **peri** (*number, optional*) -- The periodicity of the array. For example, 360 for longitude.
- **ascending** (*int, default 1*) -- Whether the array is in ascending order. 1 for ascending order, -1 for descending order.
- **above** (*boolean, default True*) -- If True, return the index of the largest item in array smaller than value. Otherwise, return the closest value.
- **dx_right** (*boolean, default True*) -- If True, darray[i] = abs(array[i+1] - array[i])

**Returns**

- **ix** (*numpy.ndarray*) -- Indexes of the reference level
- **rx** (*numpy.ndarray*) -- Non-dimensional distance to the reference level
- **dx** (*numpy.ndarray*) -- distance between the reference t level and the next one.
- **bx** (*numpy.ndarray*) -- Value of the reference level

---

## find_rel_h_naive

```python
find_rel_h_naive(lon, lat, some_x, some_y, some_dx, some_dy, CS, SN, tree)
```

Find the rel-coords in the horizontal.

very similar to find_rel_time/v
rx,ry,dx,dy are defined the same way
for example
rx = "how much to the right of the node"/"size of the cell in left-right direction"
dx = "size of the cell in left-right direction".

cs,sn is just the cos and sin of the grid orientation.
It will come in handy when we transfer vectors.

---

## find_rel_h_oceanparcel

```python
find_rel_h_oceanparcel(x, y, some_x, some_y, some_dx, some_dy, CS, SN, XG, YG, tree, tp)
```

Find the rel-coords using the rectilinear scheme.

---

## find_rel_h_rectilinear

```python
find_rel_h_rectilinear(x, y, lon, lat)
```

Find the rel-coords using the rectilinear scheme.

---

## find_rel_nearest

```python
find_rel_nearest(value, ts)
```

Find the rel-coords based on the find_ind_nearest method.

---

## find_rel_periodic

```python
find_rel_periodic(value, ts, peri)
```

Find the rel-coords based on the find_ind_periodic method.

---

## find_rel_time

```python
find_rel_time(time, ts)
```

Find the rel-coords of the temporal coords.

**Parameters**

- **time** (*numpy.ndarray*) -- 1D array for the time since 1970-01-01 in seconds.
- **ts** (*numpy.ndarray*) -- The time of model time steps also in seconds.

**Returns**

- **it** (*numpy.ndarray*) -- Indexes of the reference t level
- **rt** (*numpy.ndarray*) -- Non-dimensional distance to the reference t level
- **dt** (*numpy.ndarray*) -- distance between the reference t level and the next one.

---

## find_rel_z

```python
find_rel_z(depth, some_z, some_dz=None, dz_above_z=True)
```

Find the rel-coords of the vertical coords.

**Parameters**

- **depth** (*numpy.ndarray*) -- 1D array for the depth of interest in meters. More negative means deeper.
- **some_z** (*numpy.ndarray*) -- The depth of reference depth.
- **some_dz** (*numpy.ndarray or None*) -- dz_i = abs(z_{i+1}- z_i)
- **dz_above_z** (*Boolean*) -- Whether the dz as the distance between the depth level and a shallower one(True) or a deeper one(False)

**Returns**

- **iz** (*numpy.ndarray*) -- Indexes of the reference z level
- **rz** (*numpy.ndarray*) -- Non-dimensional distance to the reference z level
- **dz** (*numpy.ndarray*) -- distance between the reference z level and the next one.

---

## find_rx_ry_naive

```python
find_rx_ry_naive(x, y, bx, by, cs, sn, dx, dy)
```

Find the non-dimensional coords using the local cartesian scheme.

---

## find_rx_ry_oceanparcel

```python
find_rx_ry_oceanparcel(x, y, px, py)
```

Find the non-dimensional horizontal distance.

This is done using the oceanparcel scheme.

---

## get_key_by_value

```python
get_key_by_value(d, value)
```

Find one of the keys in a dictionary.

the key that correspond to the given value.

**Parameters**

- **d** (*dictionaty*) -- dictionary to lookup key from
- **value** (*object*) -- A object that has __eq__ method.

---

## local_to_latlon

```python
local_to_latlon(u, v, cs, sn)
```

Convert local vector to north-east.

---

## missing_cs_sn

```python
missing_cs_sn(ds, return_xr=False)
```

Fill in the CS,SN of a dataset.

---

## parallelpointinpolygon

```python
parallelpointinpolygon(xs, ys, poly)
```

Check if xs,ys is in the polygon, return same size boolean array.

**Parameters**

- **xs,ys** (*1D np.array*) -- the x,y locations
- **poly** (*2D np.array*) -- the location of the edge of polygon, the order matters.

---

## pointinpolygon

```python
pointinpolygon(x, y, poly)
```

Check if x,y is in the polygon.

---

## process_ecco

```python
process_ecco(ds)
```

Add more meat to ECCO dataset after the skeleton is downloaded.

---

## rel2latlon

```python
rel2latlon(rx, ry, cs, sn, dx, dy, bx, by)
```

Translate the spatial rel-coords into lat-lon-dep coords.

---

## spherical2cartesian

```python
spherical2cartesian(lat, lon, R=6371.0)
```

Convert spherical coordinates to cartesian.

**Parameters**

- **lat** (*np.array*) -- Spherical Y coordinate (latitude)
- **lon** (*np.array*) -- Spherical X coordinate (longitude)
- **R** (*scalar*) -- Earth radius in km If None, use geopy default

**Returns**

- **x** (*np.array*) -- Cartesian x coordinate
- **y** (*np.array*) -- Cartesian y coordinate
- **z** (*np.array*) -- Cartesian z coordinate

---

## to_180

```python
to_180(x, peri=360)
```

Convert any longitude scale to [-180,180).

---

## weight_f_node

```python
weight_f_node(rx, ry)
```

Assign weights to four corners.

assign weight based on the non-dimensional
coords to the four corner points.
