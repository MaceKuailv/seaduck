# seaduck.ocedata.RelCoord and subclasses

## RelCoord

```python
class RelCoord()
```

NamedTuple that also has update method.

This class is used to store the relative coordinates.
Attributes starts with "i" are indexes of the nearest grid point.
Attributes starts with "b" are value (time/dep/lat/lon) of the nearest grid point.
Attributes starts with "d" are distance between the nearest grid point and its
neighboring point in meters or seconds.
Attributes starts with "r" are the distance from the point of interest to the nearest
non-dimensionalized by the "d" variable.
"cs", "sn" are the cosine and sine of the grid orientation relative to meridian.
"face" is the face/tile the point is on, if the dataset has such a dimension.

All of those attributes should be None or 1D numpy array.

### RelCoord.create_class   *(method)*

```python
create_class(class_name, fields)
```

Create a subclass with predetermined keys.

### RelCoord.subset   *(method)*

```python
subset(self, which)
```

Create a subset of all the non-None items.

---

## HRel

```python
class HRel(*args)
```

Wrap around the horizontal rel-coords. See also RelCoord.

### HRel.create_class   *(method)*

```python
create_class(class_name, fields)
```

Create a subclass with predetermined keys.

### HRel.subset   *(method)*

```python
subset(self, which)
```

Create a subset of all the non-None items.

---

## VRel

```python
class VRel(*args)
```

Wrap around the vertical centered nearest rel-coords. See also RelCoord.

### VRel.create_class   *(method)*

```python
create_class(class_name, fields)
```

Create a subclass with predetermined keys.

### VRel.subset   *(method)*

```python
subset(self, which)
```

Create a subset of all the non-None items.

---

## VLRel

```python
class VLRel(*args)
```

Wrap around the vertical centered linear rel-coords. See also RelCoord.

### VLRel.create_class   *(method)*

```python
create_class(class_name, fields)
```

Create a subclass with predetermined keys.

### VLRel.subset   *(method)*

```python
subset(self, which)
```

Create a subset of all the non-None items.

---

## VlRel

```python
class VlRel(*args)
```

Wrap around the vertical staggered nearest rel-coords. See also RelCoord.

### VlRel.create_class   *(method)*

```python
create_class(class_name, fields)
```

Create a subclass with predetermined keys.

### VlRel.subset   *(method)*

```python
subset(self, which)
```

Create a subset of all the non-None items.

---

## VlLinRel

```python
class VlLinRel(*args)
```

Wrap around the vertical staggered linear rel-coords. See also RelCoord.

### VlLinRel.create_class   *(method)*

```python
create_class(class_name, fields)
```

Create a subclass with predetermined keys.

### VlLinRel.subset   *(method)*

```python
subset(self, which)
```

Create a subset of all the non-None items.

---

## TRel

```python
class TRel(*args)
```

Wrap around the temporal linear rel-coords. See also RelCoord.

### TRel.create_class   *(method)*

```python
create_class(class_name, fields)
```

Create a subclass with predetermined keys.

### TRel.subset   *(method)*

```python
subset(self, which)
```

Create a subset of all the non-None items.

---

## TLinRel

```python
class TLinRel(*args)
```

NamedTuple that also has update method.

This class is used to store the relative coordinates.
Attributes starts with "i" are indexes of the nearest grid point.
Attributes starts with "b" are value (time/dep/lat/lon) of the nearest grid point.
Attributes starts with "d" are distance between the nearest grid point and its
neighboring point in meters or seconds.
Attributes starts with "r" are the distance from the point of interest to the nearest
non-dimensionalized by the "d" variable.
"cs", "sn" are the cosine and sine of the grid orientation relative to meridian.
"face" is the face/tile the point is on, if the dataset has such a dimension.

All of those attributes should be None or 1D numpy array.

### TLinRel.create_class   *(method)*

```python
create_class(class_name, fields)
```

Create a subclass with predetermined keys.

### TLinRel.subset   *(method)*

```python
subset(self, which)
```

Create a subset of all the non-None items.
