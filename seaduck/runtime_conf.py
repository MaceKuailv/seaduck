rcParam = {
    "compilable": True,
    "dump_masks_to_local": False,
}

try:  # pragma: no cover
    import numba
    from numba import njit

    rcParam["compilable"] = True
except ImportError:  # pragma: no cover
    rcParam["compilable"] = False


def compileable(func):  # pragma: no cover
    """Decorate function to compile them using numba when available."""
    if rcParam["compilable"]:
        return njit(func)
    else:
        return func


def compileable_parallel(func):  # pragma: no cover
    """Decorate function to compile them (parallel) using numba when available."""
    if rcParam["compilable"]:
        return njit(func, parallel=True)
    else:
        return func


def prange(*arg):  # pragma: no cover
    """Decorate function to compile them (parallel) using numba when available."""
    if rcParam["compilable"]:
        return numba.prange(*arg)
    else:
        return range(*arg)
