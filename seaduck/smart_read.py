import numpy as np
import xarray as xr


def smart_read(da, ind, memory_chunk=3, dask_more_efficient=100):
    """Read from a xarray.DataArray given tuple indexes.

    Try to do it fast and smartly.

    Parameters
    ----------
    da: xarray.DataArray
        DataArray to read from
    ind: tuple of numpy.ndarray
        The indexes of points of interest, each element does not need to be 1D
    memory_chunk: int, default 3
        If the number of chunks needed is smaller than this, read all of them at once.
    xarray_more_efficient: int, default 100
        When the number of chunks is larger than this, and the data points are few,
        it may make sense to directly use xarray's vectorized read.

    Returns
    -------
    + values: numpy.ndarray
        The values of the points of interest. Has the same shape as the elements in ind.
    """
    the_shape = ind[0].shape
    ind = tuple(i.ravel() for i in ind)
    if len(da.dims) != len(ind):
        raise ValueError("index does not match the number of dimensions")
    if da.chunks is None or da.chunks == {}:
        npck = np.array(da)
        return npck[ind].reshape(the_shape)
    if (
        np.prod([len(i) for i in da.chunks]) <= memory_chunk
    ):  # if the number of chunks is small don't bother
        npck = np.array(da)
        return npck[ind].reshape(the_shape)
    cksz = dict(zip(da.dims, da.chunks))
    keys = list(cksz.keys())
    n = len(ind[0])
    result = np.zeros(n)

    new_dic = {}
    # typically what happens is that the first a few indexes are chunked
    # here we figure out what is the last dimension chunked.
    for i in range(len(cksz) - 1, -1, -1):
        if len(cksz[keys[i]]) > 1:
            last = i
            break

    ckbl = np.zeros((n, i + 1)).astype(int)
    # register each each dimension and the chunk they are in
    for i, k in enumerate(keys[: i + 1]):
        ix = ind[i]
        suffix = np.cumsum(cksz[k])
        new_dic[i] = suffix
        ckbl[:, i] = np.searchsorted(suffix, ix, side="right")
    # this is the time limiting step for localized long query.
    ckus, inverse = np.unique(ckbl, axis=0, return_inverse=True)
    # ckus is the individual chunks used
    if len(ckus) <= dask_more_efficient:
        # logging.debug('use smart')
        for i, k in enumerate(ckus):
            ind_str = []
            pre = []
            which = inverse == i
            for j, p in enumerate(k):
                sf = new_dic[j][p]  # the upperbound of index
                pr = sf - cksz[keys[j]][p]  # the lower bound of index
                ind_str.append(slice(pr, sf))
                pre.append(pr)
            prs = np.zeros(len(keys)).astype(int)
            prs[: last + 1] = pre
            npck = np.array(da[tuple(ind_str)])
            subind = tuple(ind[dim][which] - prs[dim] for dim in range(len(ind)))
            result[which] = npck[subind]
        return result.reshape(the_shape)
    else:
        # logging.debug('use xarray')
        xrind = tuple(xr.DataArray(dim, dims=["x"]) for dim in ind)
        return np.array(da[xrind]).reshape(the_shape)


# # def slice_data_and_shift_indexes(da, indexes_tuple):
# #     """Slice data using min/max indexes, and shift indexes."""
# #     slicers = ()
# #     for indexes, size in zip(indexes_tuple, da.shape):
# #         try:
# #             start = indexes[indexes >= 0].min()
# #         except ValueError:
# #             start = None
# #         stop = stop if (stop := indexes.max() + 1) < size else None
# #         slicers += (slice(start, stop),)
# #     indexes_tuple = tuple(
# #         indexes.ravel() - slicer.start if slicer.start else indexes.ravel()
# #         for indexes, slicer in zip(indexes_tuple, slicers)
# #     )
# #     return da.data[slicers], indexes_tuple


# def smart_read(da, indexes_tuple, dask_more_efficient=10, chunks="auto"):
#     """Read from a xarray.DataArray given a tuple indexes.

#     Try to do it fast and smartly.
#     There is a lot of improvement to be made here,
#     but this is how it is currently done.

#     The data we read is going to be unstructured but they tend to be
#     rather localized. For example, the lagrangian particles read data
#     from the same time step. Currently, using dask/xarray's unstructured
#     read does not really take advantage of the locality.
#     This function figures out which chunks stores the data, convert them
#     into numpy arrays, and then read the data from the converted ones.

#     Parameters
#     ----------
#     da: xarray.DataArray
#         DataArray to read from
#     indexes_tuple: tuple of numpy.ndarray
#         The indexes of points of interest, each element does not need to be 1D
#     dask_more_efficient: int, default 100
#         When the number of chunks is larger than this, and the data points are few,
#         it may make sense to directly use dask's vectorized read.
#     chunks: int, str, default: "auto"
#         Chunks for indexes

#     Returns
#     -------
#     + values: numpy.ndarray
#         The values of the points of interest. Has the same shape as the elements in indexes_tuple.
#     """
#     if len(indexes_tuple) != da.ndim:
#         raise ValueError(
#             "indexes_tuple does not match the number of dimensions: "
#             f"{len(indexes_tuple)} vs {da.ndim}"
#         )

#     shape = indexes_tuple[0].shape
#     size = indexes_tuple[0].size
#     if not size:
#         # This is to make the special case of reading nothing
#         # looks normal in other parts of the code.
#         return np.empty(shape)

#     data = da.data
#     # data, indexes_tuple = slice_data_and_shift_indexes(da, indexes_tuple)
#     if isinstance(data, np.ndarray):
#         return data[indexes_tuple].reshape(shape)

#     if dask.array.empty(size, chunks=chunks).numblocks[0] > 1:
#         indexes_tuple = tuple(
#             dask.array.from_array(indexes, chunks=chunks) for indexes in indexes_tuple
#         )

#     block_dict = {}
#     for block_ids in np.ndindex(*data.numblocks):
#         if len(block_dict) >= dask_more_efficient:
#             return (
#                 data.vindex[tuple(map(dask.array.compute, indexes_tuple))]
#                 .compute()
#                 .reshape(shape)
#             )

#         shifted_indexes = []
#         mask = None
#         for block_id, indexes, chunks in zip(block_ids, indexes_tuple, data.chunks):
#             shifted = indexes - sum(chunks[:block_id])
#             block_mask = (shifted >= 0) & (shifted < chunks[block_id])
#             if not (mask := block_mask if mask is None else mask & block_mask).any():
#                 break  # empty block
#             shifted_indexes.append(shifted)
#         else:
#             block_dict[block_ids] = (
#                 np.nonzero(mask),
#                 tuple(indexes[mask] for indexes in shifted_indexes),
#             )
#             if sum([len(v[0]) for v in block_dict.values()]) == size:
#                 break  # all blocks found

#     values = np.empty(size)
#     for block_ids, (values_indexes, block_indexes) in block_dict.items():
#         block_values = data.blocks[block_ids].compute()
#         values[values_indexes] = block_values[block_indexes]
#     return values.reshape(shape)
