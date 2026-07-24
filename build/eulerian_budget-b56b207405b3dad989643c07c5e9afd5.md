# seaduck.eulerian_budget

## bolus_vel_from_psi

```python
bolus_vel_from_psi(tub, xgcmgrd, psixname='GM_PsiX', psiyname='GM_PsiY')
```

Calculate bolus velocity based on its streamfunction.

---

## buffer_x_periodic

```python
buffer_x_periodic(s, lm, rm)
```


---

## buffer_x_withface

```python
buffer_x_withface(s, face, lm, rm, tp)
```

Create buffer zone in x direction for one face of an array.

**Parameters**

- **s** (*numpy.ndarray*) -- the center field, the last dimension being X, the third last dimension being face.
- **face** (*int*) -- which face to create buffer for.
- **lm** (*int*) -- the size of the margin to the left.
- **rm** (*int*) -- the size of the margin to the right.
- **tp** (*seaduck.Topology*) -- the topology object of the

---

## buffer_y_periodic

```python
buffer_y_periodic(s, lm, rm)
```


---

## buffer_y_withface

```python
buffer_y_withface(s, face, lm, rm, tp)
```

Create buffer zone in x direction for one face of an array.

**Parameters**

- **s** (*numpy.ndarray*) -- the center field, the last dimension being X, the third last dimension being face.
- **face** (*int*) -- which face to create buffer for.
- **lm** (*int*) -- the size of the margin to the bottom.
- **rm** (*int*) -- the size of the margin to the top.
- **tp** (*seaduck.Topology*) -- the topology object of the

---

## buffer_z_nearest

```python
buffer_z_nearest(s, lm, rm)
```


---

## create_ecco_grid

```python
create_ecco_grid(ds, for_outer=False)
```

Create xgcm object for an ECCO dataset.

---

## create_periodic_grid

```python
create_periodic_grid(ds)
```


---

## hor_div

```python
hor_div(tub, xgcmgrd, xfluxname, yfluxname)
```

Calculate horizontal divergence using xgcm.

**Parameters**

- **tub** (*sd.OceData or xr.Dataset*) -- The dataset to calculate data from
- **xgcmgrd** (*xgcm.Grid*) -- The Grid of the dataset
- **xfluxname, yfluxname** (*string*) -- The name of the variables corresponding to the horizontal fluxes in concentration m^3/s

---

## second_order_flux_limiter_x

```python
second_order_flux_limiter_x(s_center, u_cfl)
```

Get interpolated tracer concentration in X.

for more info, see
https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#second-order-flux-limiters

**Parameters**

- **s_center** (*numpy.ndarray*) -- the tracer field with buffer zone added (lm=2,rm=1), the last dimension being X
- **u_cfl** (*numpy.ndarray*) -- the u velocity normalized by grid size in X, in s^-1.

---

## second_order_flux_limiter_y

```python
second_order_flux_limiter_y(s_center, u_cfl)
```

Get interpolated tracer concentration in Y.

for more info, see
https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#second-order-flux-limiters

**Parameters**

- **s_center** (*numpy.ndarray*) -- the tracer field with buffer zone added (lm=2,rm=1), the last dimension being X
- **u_cfl** (*numpy.ndarray*) -- the u velocity normalized by grid size in y, in s^-1.

---

## second_order_flux_limiter_z

```python
second_order_flux_limiter_z(s_center, u_cfl)
```

Get interpolated tracer concentration in Z.

for more info, see
https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#second-order-flux-limiters

**Parameters**

- **s_center** (*numpy.ndarray*) -- the tracer field with buffer zone added (lm=2,rm=1), the last dimension being X
- **u_cfl** (*numpy.ndarray*) -- the u velocity normalized by grid size in z, in s^-1.

---

## third_order_DST_x

```python
third_order_DST_x(xbuffer, u_cfl)
```

Get interpolated tracer concentration in X.

for more info, see
https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#third-order-direct-space-time

**Parameters**

- **xbuffer** (*numpy.ndarray*) -- the tracer field with buffer zone added (lm=2,rm=1), the last dimension being X
- **u_cfl** (*numpy.ndarray*) -- the u velocity normalized by grid size in x, in s^-1.

---

## third_order_DST_y

```python
third_order_DST_y(ybuffer, u_cfl)
```

Get interpolated tracer concentration in Y.

for more info, see
https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#third-order-direct-space-time

**Parameters**

- **ybuffer** (*numpy.ndarray*) -- the tracer field with buffer zone added (lm=2,rm=1), the second last dimension being Y
- **u_cfl** (*numpy.ndarray*) -- the v velocity normalized by grid size in y, in s^-1.

---

## third_order_upwind_z

```python
third_order_upwind_z(s, w)
```

Get interpolated tracer concentration in the vertical.

for more info, see
https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#third-order-upwind-bias-advection
This function currently only work when there is no through surface flux.

**Parameters**

- **s** (*numpy.ndarray*) -- the tracer field, the first dimension being Z
- **w** (*numpy.ndarray*) -- the vertical velocity field, the first dimension being Zl

---

## total_div

```python
total_div(tub, xgcmgrd, xfluxname, yfluxname, zfluxname)
```

Calculate 3D divergence using xgcm.

**Parameters**

- **tub** (*sd.OceData or xr.Dataset*) -- The dataset to calculate data from
- **xgcmgrd** (*xgcm.Grid*) -- The Grid of the dataset
- **zfluxname** (*string*) -- The name of the variables corresponding to the vertical flux in concentration m^3/s

---

## ver_div

```python
ver_div(tub, xgcmgrd, zfluxname)
```

Calculate horizontal divergence using xgcm.

**Parameters**

- **tub** (*sd.OceData or xr.Dataset*) -- The dataset to calculate data from
- **xgcmgrd** (*xgcm.Grid*) -- The Grid of the dataset
- **xfluxname, yfluxname, zfluxname** (*string*) -- The name of the variables corresponding to the fluxes in concentration m^3/s
