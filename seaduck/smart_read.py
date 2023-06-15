import numpy as np


def smart_read(da, indexes_tuple, dask_more_efficient=100):
    """Read from a xarray.DataArray given a tuple indexes.

    Try to do it fast and smartly.

    Parameters
    ----------
    da: xarray.DataArray
        DataArray to read from
    indexes_tuple: tuple of numpy.ndarray
        The indexes of points of interest, each element does not need to be 1D
    dask_more_efficient: int, default 100
        When the number of chunks is larger than this, and the data points are few,
        it may make sense to directly use dask's vectorized read.

    Returns
    -------
    + values: numpy.ndarray
        The values of the points of interest. Has the same shape as the elements in indexes_tuple.
    """
    if len(indexes_tuple) != da.ndim:
        raise ValueError(
            "indexes_tuple does not match the number of dimensions: "
            f"{len(indexes_tuple)} vs {da.ndim}"
        )

    shape = indexes_tuple[0].shape
    size = indexes_tuple[0].size
    indexes_tuple = tuple(indexes.ravel() for indexes in indexes_tuple)

    if not da.chunks:
        return da.values[indexes_tuple].reshape(shape)
    data = da.data

    found_count = 0
    block_dict = {}
    for block_ids in np.ndindex(*data.numblocks):
        shifted_indexes = []
        mask = True
        for block_id, indexes, chunks in zip(block_ids, indexes_tuple, data.chunks):
            shifted = indexes - sum(chunks[:block_id])
            shifted_indexes.append(shifted)
            block_mask = (shifted >= 0) & (shifted < chunks[block_id])
            if not block_mask.any() or not (mask := mask & block_mask).any():
                break  # empty block
        else:
            block_dict[block_ids] = (mask, shifted_indexes)
            if len(block_dict) > dask_more_efficient:
                return data.vindex[indexes_tuple].compute().reshape(shape)

            if (found_count := found_count + mask.sum()) == size:
                break  # all blocks found

    values = np.empty(size)
    for block_ids, (mask, shifted_indexes) in block_dict.items():
        block_values = data.blocks[block_ids].compute()
        values[mask] = block_values[tuple(indexes[mask] for indexes in shifted_indexes)]
    return values.reshape(shape)
