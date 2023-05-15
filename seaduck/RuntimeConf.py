rcParam = {
    "compilable": True,
    "debug_level": "not that high",
    "dump_masks_to_local": False,
}

try: # pragma: no cover
    from numba import njit
    useJIT = True
except ImportError:# pragma: no cover
    useJIT = False
    
def compileable(func): # pragma: no cover
    if useJIT:
        return njit(func)
    else:
        return func