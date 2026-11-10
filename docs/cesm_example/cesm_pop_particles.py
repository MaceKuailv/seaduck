"""Lagrangian particles in CESM2 (POP ocean) with seaduck.

Particles are released in the Beaufort Gyre at 50 m depth and advected for a
year with the monthly-mean CESM2-WACCM piControl output.

The history files used here only contain the advective salt fluxes UES, VNS
and WTS (g/kg/s, i.e. salt transport divided by the T-cell volume) on the
east, north and top faces of each T cell. The volume transport (m^3/s) is
recovered by dividing by the salinity on the cell face,

    U_east = UES * TAREA * dz / S_east,

which is closed to within ~1e-5 (median relative residual of the volume
divergence). If UVEL/VVEL/WVEL or CMIP umo/vmo/wmo are available, use those
instead. The transport is the resolved (Eulerian-mean) flow only: the GM and
submesoscale eddy-induced transports are not included.

Units and signs: depth is negative upward in meters, w is positive upward,
time is in seconds since 0000-01-01 (noleap), and longitude is degrees east.
"""

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

import seaduck as sd  # noqa: E402

DATA = "/vast/thaine1/cwill353/CESM2-WACCM"
CASE = "b.e21.BW1850.f09_g17.CMIP6-piControl.001.pop.h"
PERIOD = "000101-009912"
N_MONTH = 13


def open_pop(name):
    """Open the first N_MONTH months of one POP history variable."""
    path = f"{DATA}/{name}/{CASE}.{name}.{PERIOD}.nc"
    return xr.open_dataset(path, decode_times=False).isel(time=slice(N_MONTH))


def face_mean(a, b):
    """Salinity on the face between two cells, one-sided next to land."""
    return 0.5 * (a.fillna(b) + b.fillna(a))


# ------------------------------------------------------------------
# Volume transport on the cell faces, still in POP's layout
# ------------------------------------------------------------------
ds = open_pop("SALT")
for name in ["UES", "VNS", "WTS"]:
    ds[name] = open_pop(name)[name]

salt = ds["SALT"]
vol = ds["TAREA"] * ds["dz"] * 1e-6  # m^3
salt_top = face_mean(salt, salt.shift(z_t=1))
salt_top = salt_top.drop_vars("z_t").rename(z_t="z_w_top")
vol_top = vol.drop_vars("z_t").rename(z_t="z_w_top")

for flux, transport, s_face, v in [
    ("UES", "utrans", face_mean(salt, salt.roll(nlon=-1)), vol),
    ("VNS", "vtrans", face_mean(salt, salt.shift(nlat=-1)), vol),
    ("WTS", "wtrans", salt_top, vol_top),
]:
    ds[transport] = (ds[flux] * v / s_face).fillna(0).astype("float32")
    # process_pop uses grid_loc to put the variable on the right face
    ds[transport].attrs = {"grid_loc": ds[flux].attrs["grid_loc"], "units": "m^3/s"}

ds = ds.drop_vars(["UES", "VNS", "WTS", "SALT"])
ds = sd.utils.process_pop(ds).load()

# ------------------------------------------------------------------
# Particle simulation
# ------------------------------------------------------------------
od = sd.OceData(ds)
print("topology:", od.tp.typ, "| horizontal scheme:", od.readiness["h"])

lon, lat = np.meshgrid(np.linspace(-160, -130, 31), np.linspace(72, 79, 15))
x = lon.ravel()
y = lat.ravel()
z = np.full_like(x, -50.0)  # m, negative upward
t = np.full_like(x, od.ts[0])  # s, middle of the first month
tf = od.ts[-1]

p = sd.Particle(
    x=x,
    y=y,
    z=z,
    t=t,
    data=od,
    uname="utrans",
    vname="vtrans",
    wname="wtrans",
    transport=True,
)
# remove particles on land.
p = p.subset(np.array(od["maskC"])[p.izl_lin - 1, p.iy, p.ix])

normal_stops = np.linspace(t[0], tf, 4 * (N_MONTH - 1) + 1)
stops, raw = p.to_list_of_time(normal_stops=normal_stops)

lons = np.array([pt.lon for pt in raw]).T
lats = np.array([pt.lat for pt in raw]).T
deps = np.array([pt.dep for pt in raw]).T

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
days = (np.array(stops) - stops[0]) / 86400
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-150))
ax.plot(lons.T, lats.T, color="darkblue", lw=0.4, transform=ccrs.PlateCarree())
ax.plot(lons[:, 0], lats[:, 0], "k.", ms=2, transform=ccrs.PlateCarree())
ax.plot(lons[:, -1], lats[:, -1], "r.", ms=2, transform=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-180, -100, 65, 85], ccrs.PlateCarree())
ax.set_title(
    f"CESM2-WACCM particles released at 50 m, {days[-1]:.0f} days\n"
    "black: start, red: end"
)
fig.savefig("cesm_pop_trajectories.png", dpi=200, bbox_inches="tight")

print(
    f"{p.N} particles, depth range at the end: {deps[:, -1].min():.1f} to "
    f"{deps[:, -1].max():.1f} m"
)
