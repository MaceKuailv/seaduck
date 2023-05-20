import functools
import os

import numpy as np
import pooch
import xarray as xr

POOCH = pooch.create(
    path=pooch.os_cache("seaduck"), base_url="doi:10.5281/zenodo.7949168", registry=None
)
POOCH.load_registry_from_doi()  # Automatically populate the registry
POOCH_FETCH_KWARGS = {"progressbar": True}


def process_ecco(ds):
    rand1 = np.random.random((50, 13, 90, 90))
    rand2 = np.random.random((50, 13, 90, 90))
    rand3 = np.random.random((50, 13, 90, 90))
    ds["UVELMASS"] = xr.DataArray(
        np.stack([rand1, rand2, rand3], axis=0), dims=("time", "Z", "face", "Y", "Xp1")
    )
    ds["UVELMASS"][0] = ds.UVELMASS1
    ds["UVELMASS"][1] = ds.UVELMASS1 * rand1
    ds["UVELMASS"][2] = ds.UVELMASS1 * rand2

    ds["WVELMASS"] = xr.DataArray(
        np.stack([rand1, rand2, rand3], axis=0), dims=("time", "Zl", "face", "Y", "X")
    )
    ds["WVELMASS"][0] = ds.WVELMASS1
    ds["WVELMASS"][1] = ds.WVELMASS1 * rand1
    ds["WVELMASS"][2] = ds.WVELMASS1 * rand2

    ds["VVELMASS"] = xr.DataArray(
        np.stack([rand1, rand2, rand3], axis=0), dims=("time", "Z", "face", "Yp1", "X")
    )
    ds["VVELMASS"][0] = ds.VVELMASS1
    ds["VVELMASS"][1] = ds.VVELMASS1 * rand1
    ds["VVELMASS"][2] = ds.VVELMASS1 * rand2

    ds["SALT"] = xr.DataArray(
        np.stack([rand1, rand2, rand3], axis=0), dims=("time", "Z", "face", "Y", "X")
    )
    ds["SALT_snap"] = xr.DataArray(
        np.stack([rand3, rand1], axis=0), dims=("time_midp", "Z", "face", "Y", "X")
    )
    ds["ETAN"] = xr.DataArray(rand1[:3], dims=("time", "face", "Y", "X"))
    ds["ETAN_snap"] = xr.DataArray(rand3[:2], dims=("time_midp", "face", "Y", "X"))
    return ds


@functools.cache
def get_dataset(name):
    fnames = POOCH.fetch(f"{name}.tar.gz", pooch.Untar(), **POOCH_FETCH_KWARGS)
    ds = xr.open_zarr(os.path.commonpath(fnames))
    if name == "ecco":
        return process_ecco(ds)
    return ds
