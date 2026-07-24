# seaduck.smart_read

## slice_data_and_shift_indexes

```python
slice_data_and_shift_indexes(da, indexes_tuple)
```

Slice data using min/max indexes, and shift indexes.

---

## smart_read

```python
smart_read(da, indexes_tuple, dask_more_efficient=10, chunks='auto')
```

Read from a xarray.DataArray given a tuple indexes.

Try to do it fast and smartly.
There is a lot of improvement to be made here,
but this is how it is currently done.

The data we read is going to be unstructured but they tend to be
rather localized. For example, the lagrangian particles read data
from the same time step. Currently, using dask/xarray's unstructured
read does not really take advantage of the locality.
This function figures out which chunks stores the data, convert them
into numpy arrays, and then read the data from the converted ones.

**Parameters**

- **da** (*xarray.DataArray*) -- DataArray to read from
- **indexes_tuple** (*tuple of numpy.ndarray*) -- The indexes of points of interest, each element does not need to be 1D
- **dask_more_efficient** (*int, default 100*) -- When the number of chunks is larger than this, and the data points are few, it may make sense to directly use dask's vectorized read.
- **chunks** (*int, str, default: "auto"*) -- Chunks for indexes

**Returns**

- **+ values** (*numpy.ndarray*) -- The values of the points of interest. Has the same shape as the elements in indexes_tuple.
