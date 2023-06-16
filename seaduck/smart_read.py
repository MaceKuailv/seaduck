from itertools import accumulate

import numpy as np


def smart_read(da, indexes_tuple, dask_more_efficient=100):
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

    largest_indexes_of_chunk = []
    for chunks in data.chunks:
        largest_indexes_of_chunk.append(list(accumulate(chunks)))

    smallest_indexes_of_dimension = []
    for indexes in indexes_tuple:
        smallest_indexes_of_dimension.append(np.min(indexes))

    for block_ids in np.ndindex(*data.numblocks):
        if not reach_first_entry:
            for block_id, large_index, small_ind in zip(
                block_ids, largest_indexes_of_chunk, smallest_indexes_of_dimension
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
