import dask.array
import numpy as np


def slice_data_and_shift_indexes(da, indexes_tuple):
    """Slice data using min/max indexes, and shift indexes."""
    slicers = ()
    for indexes, size in zip(indexes_tuple, da.shape):
        if (indexes < 0).any():
            slicers += (slice(None),)
        else:
            start = indexes[indexes >= 0].min()
            stop = stop if (stop := indexes.max() + 1) < size else None
            slicers += (slice(start, stop),)
    indexes_tuple = tuple(
        indexes.ravel() - slicer.start if slicer.start else indexes.ravel()
        for indexes, slicer in zip(indexes_tuple, slicers)
    )
    return da.data[slicers], indexes_tuple


def smart_read(da, indexes_tuple, dask_more_efficient=10, chunks="auto"):
    """Read from a xarray.DataArray given a tuple indexes.

    Try to do it fast and smartly.
    There is a lot of improvement to be made here,
    but this is how it is currently done.

    The data we read is going to be unstructured but they tend to be
    rather localized. For example, the lagrangian particles read data
    from the same time step. Currently, using dask/xarray's unstructured
    read does not really take advantage of the locality.
    This function figures out which chunks stores the data, convert them
    into numpy arrays, and then read the data from the converted ones.

    Parameters
    ----------
    da: xarray.DataArray
        DataArray to read from
    indexes_tuple: tuple of numpy.ndarray
        The indexes of points of interest, each element does not need to be 1D
    dask_more_efficient: int, default 100
        When the number of chunks is larger than this, and the data points are few,
        it may make sense to directly use dask's vectorized read.
    chunks: int, str, default: "auto"
        Chunks for indexes

    Returns
    -------
    + values: numpy.ndarray
        The values of the points of interest. Has the same shape as the elements in indexes_tuple.
    """
    num_dims = len(indexes_tuple)
    if num_dims != da.ndim:
        raise ValueError(
            "indexes_tuple does not match the number of dimensions: "
            f"{len(indexes_tuple)} vs {da.ndim}"
        )

    shape = indexes_tuple[0].shape
    size = indexes_tuple[0].size
    indexes_tuple = tuple(index.ravel() for index in indexes_tuple)
    if not size:
        # This is to make the special case of reading nothing
        # looks normal in other parts of the code.
        return np.empty(shape)

    # data = da.data
    data, indexes_tuple = slice_data_and_shift_indexes(da, indexes_tuple)
    if isinstance(data, np.ndarray):
        return data[indexes_tuple].reshape(shape)

    if dask.array.empty(size, chunks=chunks).numblocks[0] > 1:
        indexes_tuple = tuple(
            dask.array.from_array(indexes, chunks=chunks) for indexes in indexes_tuple
        )

    block_dict = {}
    for block_ids in np.ndindex(*data.numblocks):
        if len(block_dict) >= dask_more_efficient:
            return (
                data.vindex[tuple(map(dask.array.compute, indexes_tuple))]
                .compute()
                .reshape(shape)
            )

        shifted_indexes = []
        mask = None
        for block_id, indexes, chunks in zip(block_ids, indexes_tuple, data.chunks):
            shifted = indexes - sum(chunks[:block_id])
            block_mask = (shifted >= 0) & (shifted < chunks[block_id])
            if not (mask := block_mask if mask is None else mask & block_mask).any():
                break  # empty block
            shifted_indexes.append(shifted)
        else:
            block_dict[block_ids] = (
                np.nonzero(mask),
                tuple(indexes[mask] for indexes in shifted_indexes),
            )
            if sum([len(v[0]) for v in block_dict.values()]) == size:
                break  # all blocks found

    values = np.empty(size)
    for block_ids, (values_indexes, block_indexes) in block_dict.items():
        block_values = data.blocks[block_ids].compute()
        values[values_indexes] = block_values[block_indexes]
    return values.reshape(shape)
