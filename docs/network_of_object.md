# Learn about `seaduck` core objects and functions

Another good way to learn about the package is to learn how different objects are used and related to each other.

| Object Name (and link to API reference)                   | Main functionality                                           | Example                                                      |
| --------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`seaduck.OceData`](api_reference/apiref_OceData.rst)     | Interface to data, translate between lat-lon and indices.    | [AVISO](notebook/AVISO.ipynb)                                |
| [`seaduck.Topology`](api_reference/apiref_topology.rst)   | Describe the how the grids are connected. Similar to `xgcm`. | [topology tutorial](notebook/topology_tutorial.ipynb)        |
| [`seaduck.KnW`](api_reference/apiref_kernelNweight.rst)   | Define what interpolation/derivative to perform.             | [ECCO](notebook/global_ECCO.ipynb)                           |
| [`seaduck.Position`](api_reference/apiref_eulerian.rst)   | Interpolate at Eulerian positions.                           | [horizontal stream function conservation](idealize_test/hor_stream) |
| [`seaduck.Particle`](api_reference/apiref_lagrangian.rst) | Perform Lagrangian particle simulation. Sub class of `seaduck.Position` | [Regional simulation](sciserver_notebooks/IGP.md)            |
| [`seaduck.OceInterp`](api_reference/apiref_OceInterp.rst) | Uniform interface to Lagrangian and Eulerian operations.     | [one minute guide](one_min_guide.ipynb), [ECCO](notebook/global_ECCO.ipynb)   |
