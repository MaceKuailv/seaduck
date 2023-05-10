import numpy as np
import pytest
import xarray as xr

import seaduck.utils as _u

Datadir = "tests/Data/"
ecco = xr.open_zarr(Datadir + "small_ecco")

curv = xr.open_dataset("{}MITgcm_curv_nc.nc" "".format(Datadir))

rect = xr.open_dataset("{}MITgcm_rect_nc.nc" "".format(Datadir))

_u.missing_cs_sn(curv)
_u.missing_cs_sn(rect)