try:  # pragma: no cover
    import xgcm
except ImportError:  # pragma: no cover
    pass


def _raise_if_no_xgcm():
    try:
        import xgcm

        xgcm
    except ImportError:
        raise ImportError(
            "The python package xgcm is needed."
            "You can install it with:"
            "conda install -c xgcm"
        )


def create_ecco_grid(ds):
    face_connections = {
        "face": {
            0: {
                "X": ((12, "Y", False), (3, "X", False)), 
                "Y": (None, (1, "Y", False))
            },
            1: {
                "X": ((11, "Y", False), (4, "X", False)),
                "Y": ((0, "Y", False), (2, "Y", False)),
            },
            2: {
                "X": ((10, "Y", False), (5, "X", False)),
                "Y": ((1, "Y", False), (6, "X", False)),
            },
            3: {
                "X": ((0, "X", False), (9, "Y", False)), 
                "Y": (None, (4, "Y", False))
            },
            4: {
                "X": ((1, "X", False), (8, "Y", False)),
                "Y": ((3, "Y", False), (5, "Y", False)),
            },
            5: {
                "X": ((2, "X", False), (7, "Y", False)),
                "Y": ((4, "Y", False), (6, "Y", False)),
            },
            6: {
                "X": ((2, "Y", False), (7, "X", False)),
                "Y": ((5, "Y", False), (10, "X", False)),
            },
            7: {
                "X": ((6, "X", False), (8, "X", False)),
                "Y": ((5, "X", False), (10, "Y", False)),
            },
            8: {
                "X": ((7, "X", False), (9, "X", False)),
                "Y": ((4, "X", False), (11, "Y", False)),
            },
            9: {
                "X": ((8, "X", False), None), 
                "Y": ((3, "X", False), (12, "Y", False))
            },
            10: {
                "X": ((6, "Y", False), (11, "X", False)),
                "Y": ((7, "Y", False), (2, "X", False)),
            },
            11: {
                "X": ((10, "X", False), (12, "X", False)),
                "Y": ((8, "Y", False), (1, "X", False)),
            },
            12: {
                "X": ((11, "X", False), None),
                "Y": ((9, "Y", False), (0, "X", False)),
            },
        }
    }

    grid = xgcm.Grid(
        ds,
        periodic=False,
        face_connections=face_connections,
        coords={
            "X": {"center": "X", "left": "Xp1"},
            "Y": {"center": "Y", "left": "Yp1"},
            "Z": {"center": "Z", "left": "Zl", "outer": "Zp1", "right": "Zu"},
            "time": {"center": "time", "inner": "time_midp"},
        },
    )
    return grid


def hor_div(tub, grid, xfluxname, yfluxname):
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
        {"X": tub[xfluxname], "Y": tub[yfluxname]}, boundary="fill", fill_value=0.0
    )
    x_diff = xy_diff["X"]
    y_diff = xy_diff["Y"]
    hConv = -(x_diff + y_diff) / tub["Vol"]
    return hConv


def ver_div(tub, grid, zfluxname):
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
    vConv = grid.diff(tub[zfluxname], "Z", boundary="fill", fill_value=0.0) / tub["Vol"]
    return vConv
