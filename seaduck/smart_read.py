from itertools import accumulate

import numpy as np


def smart_read(da, indexes_tuple, dask_more_efficient=100, dense=1e7):
    """Read from a xarray.DataArray given a tuple indexes.

    Try to do it fast and smartly.
    There is a lot of improvement to be made here,
    but this is how it is currently done.

    The data we read is going to be unstructured but they tend to be
    rather localized. For example, the lagrangian particles read data
    from the same time step.
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
    reach_first_entry = False
    block_dict = {}

    min_indexes_of_dim = []
    for indexes in indexes_tuple:
        min_indexes_of_dim.append(np.min(indexes))

    if dense:
        max_indexes_of_dim = []
        shifted_indexes = []
        for idim, indexes in enumerate(indexes_tuple):
            max_indexes_of_dim.append(np.max(indexes))
            shifted_indexes.append(indexes - min_indexes_of_dim[idim])
        minmax = zip(min_indexes_of_dim, max_indexes_of_dim)
        slice_tuple = tuple(slice(mn, mx + 1) for mn, mx in minmax)
        dense_block_size = np.prod([mx - mn + 1 for mn, mx in minmax])
        if dense_block_size <= dense:
            dense_block = data[slice_tuple].compute()
            values = dense_block[tuple(shifted_indexes)]
            return values.reshape(shape)

    max_indexes_of_chunk = []
    for chunks in data.chunks:
        max_indexes_of_chunk.append(list(accumulate(chunks)))

    for block_ids in np.ndindex(*data.numblocks):
        if not reach_first_entry:
            for block_id, large_index, small_ind in zip(
                block_ids, max_indexes_of_chunk, min_indexes_of_dim
            ):
                if large_index[block_id] < small_ind:
                    break
            else:
                reach_first_entry = True

        shifted_indexes = []
        mask = True
        for block_id, indexes, chunks in zip(block_ids, indexes_tuple, data.chunks):
            shifted = indexes - sum(chunks[:block_id])
            block_mask = (shifted >= 0) & (shifted < chunks[block_id])
            if not block_mask.any() or not (mask := mask & block_mask).any():
                break  # empty block
            shifted_indexes.append(shifted)
        else:
            block_dict[block_ids] = (
                np.argwhere(mask).squeeze(),
                tuple(indexes[mask] for indexes in shifted_indexes),
            )
            if len(block_dict) >= dask_more_efficient:
                return data.vindex[indexes_tuple].compute().reshape(shape)

            if (found_count := found_count + mask.sum()) == size:
                break  # all blocks found

    values = np.empty(size)
    for block_ids, (values_indexes, block_indexes) in block_dict.items():
        block_values = data.blocks[block_ids].compute()
        values[values_indexes] = block_values[block_indexes]
    return values.reshape(shape)
