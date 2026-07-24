# seaduck.kernel_weight

## KnW

```python
class KnW(kernel=array([[ 0,  0],
       [ 0,  1],
       [ 0,  2],
       [ 0, -1],
       [ 0, -2],
       [-1,  0],
       [-2,  0],
       [ 1,  0],
       [ 2,  0]]), inheritance='auto', hkernel='interp', vkernel='nearest', tkernel='nearest', h_order=0, ignore_mask=False)
```

Kernel object.

A class that describes anything about the
interpolation/derivative kernel to be used.

**Parameters**

- **kernel** (*numpy.ndarray*) -- The largest horizontal kernel to be used
- **inheritance** (*list*) -- The inheritance sequence of the kernels
- **hkernel** (*str*) -- What to do in the horizontal direction 'interp', 'dx', or 'dy'?
- **tkernel** (*str*) -- What kind of operation to do in the temporal dimension: 'linear', 'nearest' interpolation, or 'dt'
- **vkernel** (*str*) -- What kind of operation to do in the vertical: 'linear', 'nearest' interpolation, or 'dz'
- **h_order** (*int*) -- How many derivative to take in the horizontal direction. Zero for pure interpolation
- **ignore_mask** (*bool*) -- Whether to diregard the masking of the dataset. You can select True if there is no inheritance, or if performance is a big concern.

### KnW.get_weight   *(method)*

```python
get_weight(self, rx, ry, rz=0, rt=0, pk4d=None, bottom_scheme='no flux')
```

Return the weight of values given particle rel-coords.

**Parameters**

- **rx,ry,rz,rt** (*numpy.ndarray*) -- 1D array of non-dimensional particle positions
- **pk4d** (*list*) -- A mapping on which points should use which kernel.
- **bottom_scheme** (*str*) -- Whether to assume there is a ghost point with same value at the bottom boundary. Choose None for vertical flux, 'no flux' for most other cases.

**Returns**

- **weight** (*numpy.ndarray*) -- The weight of interpolation/derivative for the points with shape (N,M), M is the num of node in the largest kernel.

### KnW.same_hsize   *(method)*

```python
same_hsize(self, other)
```

Return True if 2 KnW object has the same horizontal size.

### KnW.same_size   *(method)*

```python
same_size(self, other)
```

Return True if 2 KnW object has the same 4D size.

### KnW.size_hash   *(method)*

```python
size_hash(self)
```

Produce a hash value based on the 4D size of the KnW object.

---

## find_which_points_for_each_kernel

```python
find_which_points_for_each_kernel(masked, inheritance='default')
```

Find which kernel to use at each point.

masked is going to be a n*m array,
where n is the number of points of interest.
m is the size of the largest kernel.

inheritance defines the shape of smaller kernels.
say
inheritance = [
[0,1,2,3,4],
[0,1],
[0]
]
it means that the largest kernel have all 5 nodes
the second kernel only contain the first and second node,
and the last one only have the nearest neighbor.

if a row of matrix looks like [1,1,1,1,1],
the index of the row will be in the first element(list) of the return
variable.

if a row of matrix looks like [1,1,1,1,0],
although it fits both 2nd and 3rd kernel, 2nd has priority,
so the index will be in the 2nd element of the return pk.

if a row looks like [0,0,0,0,0],
none of the kernel can fit it, so the index will not be in the
return

---

## get_func

```python
get_func(kernel, **kwargs)
```

Return functions that compute weights.

Similar to the kernel_weight function,
the only difference is that this function can
read existing functions that is cached.
See _get_func_from_hashable

---

## get_weight_cascade

```python
get_weight_cascade(rx, ry, pk, kernel_large=array([[ 0,  0],
       [ 0,  1],
       [ 0,  2],
       [ 0, -1],
       [ 0, -2],
       [-1,  0],
       [-2,  0],
       [ 1,  0],
       [ 2,  0]]), inheritance=[[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 5, 7, 8], [0, 1, 3, 5, 7], [0]], funcs=[<function kernel_weight_x.<locals>.the_interp_func at 0x7f89f0ba6160>, <function kernel_weight_x.<locals>.the_interp_func at 0x7f89f0ba63e0>, <function kernel_weight_x.<locals>.the_interp_func at 0x7f89f0ba6480>, <function kernel_weight_x.<locals>.the_y_maxorder_func at 0x7f89f0ba65c0>])
```

Compute the weight.

apply the corresponding functions that was figured out in
find_which_points_for_each_kernel

**Parameters**

- **rx,ry** (*numpy.ndarray*) -- 1D array with length N of non-dimensional relative horizontal position to the nearest node
- **kernel_large** (*numpy.ndarray*) -- A numpy kernel of shape (M,2) that contains all the kernels needed.
- **inheritance** (*list of list(s)*) -- The inheritance sequence when some of the node is masked.
- **funcs** (*list of compileable functions*) -- The weight function of each kernel in the inheritance sequence.

**Returns**

- **+ weight** (*numpy.ndarray*) -- The horizontal weight of interpolation/derivative for the points with shape (N,M)

---

## kernel_weight

```python
kernel_weight(kernel, kernel_type='interp', order=0)
```

Return a function that compute weights.

A wrapper around kernel_weight_x and kernel_weight_s.
Return the function that calculate the interpolation/derivative weight
of a  Lagrangian kernel.

**Parameters**

- **kernel** (*np.ndarray*) -- 2D array with shape (n,2), where n is the number of nodes. It need to either shape like a rectangle or a cross
- **kernel_type** (*str*) -- "interp" (default): Use both x y direction for interpolation, implies that order = 0 "dx": Use only x direction for interpolation/derivative "dy": Use only y direction for interpolation/derivative
- **order** (*int*) -- The order of derivatives. Zero for interpolation.

**Returns**

- **func(rx,ry)** (*compilable function*) -- function to calculate the hotizontal interpolation/derivative weight

---

## kernel_weight_s

```python
kernel_weight_s(kernel, xorder=0, yorder=0)
```

Return the function that calculate the interpolation/derivative weight.

input needs to be a rectangle-shaped (that's where x is coming from)
Lagrangian kernel.

**Parameters**

- **kernel** (*np.ndarray*) -- 2D array with shape (n,2), where n is the number of nodes. It has to be shaped like a rectangle
- **xorder** (*int*) -- The order of derivatives in the x direction. Zero for interpolation.
- **yorder** (*int*) -- The order of derivatives in the y direction. Zero for interpolation.

**Returns**

- **func(rx,ry)** (*compilable function*) -- function to calculate the hotizontal interpolation/derivative weight

---

## kernel_weight_x

```python
kernel_weight_x(kernel, kernel_type='interp', order=0)
```

Return the function that calculate the interpolation/derivative weight.

input needs to be a cross-shaped (that's where x is coming from) Lagrangian kernel.

If you don't want to know what is going on under the hood.
it's totally fine.

all of the following is a bit hard to understand.
The k th (k>=0) derivative of the lagrangian polynomial is
$$
w_j= \frac{\Sigma_{i\neq j} Pi_{i<m-1-k} (x-x_i)}{\Pi_{i\neq j} (x_j - x_i)}
$$
for example: if the points are [-1,0,1] for point 0
k = 0: w = (x-1)(x+1)/(0-1)(0+1)
k = 1: w = [(x+1)+(x-1)]/(0-1)(0+1)

for a cross shape kernel:
f(rx,ry) = f_x(rx) + f_y(ry) - f(0,0)

**Parameters**

- **kernel** (*np.ndarray*) -- 2D array with shape (n,2), where n is the number of nodes. It has to be shaped like a cross
- **kernel_type** (*str*) -- "interp" (default): Use both x y direction for interpolation, implies that order = 0 "x": Use only x direction for interpolation/derivative "y": Use only y direction for interpolation/derivative
- **order** (*int*) -- The order of derivatives. Zero for interpolation.

**Returns**

- **func(rx,ry)** (*compilable function*) -- function to calculate the hotizontal interpolation/derivative weight

---

## show_kernels

```python
show_kernels(kernels=[array([[ 0,  0],
       [ 0,  1],
       [ 0,  2],
       [ 0, -1],
       [ 0, -2],
       [-1,  0],
       [-2,  0],
       [ 1,  0],
       [ 2,  0]]), array([[ 0,  0],
       [ 0,  1],
       [ 0,  2],
       [ 0, -1],
       [-1,  0],
       [ 1,  0],
       [ 2,  0]]), array([[ 0,  0],
       [ 0,  1],
       [ 0, -1],
       [-1,  0],
       [ 1,  0]]), array([[0, 0]])])
```

Plot a small scatter plot of the shape of a list of kernel.

**Parameters**

- **kernels** (*list of numpy.ndarray*) -- Each of the element is a (n,2) shaped array, where n is the number of element in the kernel.
