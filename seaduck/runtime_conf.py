rcParam = {
    "compilable": True,
    "dump_masks_to_local": False,
}

try:  # pragma: no cover
    from numba import njit

    rcParam["compilable"] = True
except ImportError:  # pragma: no cover
    rcParam["compilable"] = False


def compileable(func):  # pragma: no cover
    if rcParam["compilable"]:
        return njit(func)
    else:
        return func
