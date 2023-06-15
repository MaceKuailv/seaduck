import numpy as np


def smart_read(da, indexes_tuple):
    """Read from a xarray.DataArray given tuple indexes.

    Try to do it fast and smartly.

    Parameters
    ----------
    da: xarray.DataArray
        DataArray to read from
    indexes_tuple: tuple of numpy.ndarray
        The indexes of points of interest, each element does not need to be 1D

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
    indexes_tuple = tuple(indexes.ravel() for indexes in indexes_tuple)
    if da.chunks:
        values = np.empty(indexes_tuple[0].shape)
        count = 0
        for block_ids in np.ndindex(*da.data.numblocks):
            shifted_indexes = []
            mask = True
            for block_id, indexes, chunk in zip(
                block_ids, indexes_tuple, da.data.chunks
            ):
                shifted = indexes - sum(chunk[:block_id])
                shifted_indexes.append(shifted)
                block_mask = (shifted >= 0) & (shifted < chunk[block_id])
                if not block_mask.any() or not (mask := mask & block_mask).any():
                    break
            else:
                block_values = da.data.blocks[block_ids].compute()
                values[mask] = block_values[
                    tuple(indexes[mask] for indexes in shifted_indexes)
                ]
                if (count := count + mask.sum()) >= values.size:
                    break
    else:
        values = da.values[indexes_tuple]
    return values.reshape(shape)
