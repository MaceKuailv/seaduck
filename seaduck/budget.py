import xarray as xr

try: # pragma: no cover
    import xgcm
except ImportError: # pragma: no cover
    pass

def _raise_if_no_xgcm():
    try:
        import xgcm
    except ImportError:
        raise ImportError(
            'The python package xgcm is needed.'
            'You can install it with:'
            'conda install -c xgcm'
        )


def hor_div(tub,grid,xfluxname,vfluxname):
    """Calculate horizontal divergence using xgcm.

    Parameters
    ----------
    tub: sd.OceData or xr.Dataset
        The dataset to calculate data from
    grid: xgcm.Grid
        The Grid of the dataset
    xfluxname, yfluxname: string
        The name of the variables corresponding to the horizontal fluxes
        in concentration m^3/s
    """
    try:
        tub["Vol"]
    except KeyError:
        tub._add_missing_vol()
    xy_diff = grid.diff_2d_vector(
        {'X' : tub[xfluxname], 'Y' : tub[yfluxname]}, 
        boundary = 'fill', fill_value=0.0
    )
    x_diff = xy_diff['X']
    y_diff = xy_diff['Y']
    hConv = (-(x_diff + y_diff)/tub['Vol'])
    return hConv

def ver_div(tub,grid,zfluxname):
    """Calculate horizontal divergence using xgcm.

    Parameters
    ----------
    tub: sd.OceData or xr.Dataset
        The dataset to calculate data from
    grid: xgcm.Grid
        The Grid of the dataset
    zfluxname: string
        The name of the variables corresponding to the vertical flux
        in concentration m^3/s
    """
    try:
        tub["Vol"]
    except KeyError:
        tub._add_missing_vol()
    vConv = (grid.diff(tub[zfluxname], 'Z', boundary='fill', fill_value=0.0)/tub['Vol'])
    return vConv