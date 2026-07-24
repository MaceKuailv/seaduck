# seaduck.OceData

## OceData

```python
class OceData(data, alias=None, memory_limit=10000000.0)
```

Ocean dataset.

Class that enables variable aliasing, topology extraction, and 4-D translation
between latitude-longitude-depth-time grid and relative description.

**Parameters**

- **data** (*xarray.Dataset*) -- The dataset to extract grid information, create cKD tree, and Topology object on.
- **alias** (*dict, None, or 'auto'*) -- 1. dict: Map the variable used by this package (key) to that used by the dataset (value). 2. None (default): Do not apply alias. 3. 'auto' (Not implemented): Automatically generate a list for the alias.

### OceData.check_readiness   *(method)*

```python
check_readiness(self)
```

Return readiness.

Function that checks what kind of interpolation is supported.

**Returns**

- **readiness** (*dict*) -- 1. h: The scheme of horizontal interpolation to be used, including oceanparcel, local_cartesian, and 'rectilinear'. 2. Z: Whether the dataset has a vertical dimension at the center points. 3. Zl: Whether the dataset has a vertical dimension at staggered points (vertical velocity). 4. time: Whether the dataset has a temporal dimension.

### OceData.show_alias   *(method)*

```python
show_alias(self)
```

Print out the renaming in a nice pd.DataFrame format.

**Returns**

- **pretty_alias** (*pandas.DataFrame*) -- Carry the same information as self.alias, look a little nicer.
