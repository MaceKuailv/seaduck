# Learn about `seaduck` core objects and functions

Another good way to learn about the package is to learn how different objects are used and related to each other.

| Object Name (and link to API reference) | Main functionality | Example |
| --------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`seaduck.OceData`](api/OceData.md) | Interface to data, translate between lat-lon and indices. | [AVISO](notebook/AVISO.ipynb) |
| [`seaduck.Topology`](api/topology.md) | Describe the how the grids are connected. Similar to `xgcm`. | [topology tutorial](notebook/topology_tutorial.ipynb) |
| [`seaduck.KnW`](api/kernelNweight.md) | Define what interpolation/derivative to perform. | [ECCO](notebook/global_ECCO.ipynb) |
| [`seaduck.Position`](api/eulerian.md) | Interpolate at Eulerian positions. | [Interpolate in a fjord](sciserver_notebooks/KangerFjord.md) |
| [`seaduck.Particle`](api/lagrangian.md) | Perform Lagrangian particle simulation. Sub class of `seaduck.Position` | [Regional simulation](sciserver_notebooks/IGPwinter.md) |
| [`seaduck.OceInterp`](api/OceInterp.md) | Uniform interface to Lagrangian and Eulerian operations. | [one minute guide](one_min_guide.ipynb), [ECCO](notebook/global_ECCO.ipynb) |
