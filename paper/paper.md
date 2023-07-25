______________________________________________________________________

title: 'Seaduck: A python package for Eulerian and Lagrangian interpolation on ocean datasets'
tags:

- Python
- oceanography
- interpolation
- lagarangian particle
  authors:
- name: Wenrui Jiang
  orcid: 0009-0000-7319-9469
  equal-contrib: false
  affiliation: 1
- name: Thomas W. N. Haine
  orcid: 0000-0001-8231-2419
  affiliation: 1
- name: Mattia Almansi
  orcid: 0000-0001-6849-3647
  affiliation: 2
  affiliations:
- name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
  index: 1
- name: B-Open Solutions s.r.l, Italy
  index: 2
  date: 25 July 2023
  bibliography: paper/paper.bib

______________________________________________________________________

# Summary

Numerical simulations of the Earth's oceans are becoming more realistic and sophisticated. However, their complex layout and shear volume make it difficult for researchers to access and understand these data. Additionally, most ocean models,  mostly finite-volume models, compute and calculate spatially-integrated properties, such as grid-cell averaged temperature or wall-integrated mass flux. On the other hand, in-situ oceanographic observations are effectively collected at points in space and time. This fundamental difference makes the comparison between observations and results from numerical simulation difficult.

In this work, we present seaduck, a Python package that can perform both Eulerian and Lagrangian interpolation on generic ocean datasets with good performance and scalability. This package accesses numerical datasets from the perspective of a space-time point. It automatically navigates complex layouts of datasets and makes almost everything defined at those points available to the user. Similar to fixed and mobile observational platforms, the points can be defined from both the Eulerian and the Lagrangian perspectives. Or in other words, users can  access data from both physical and label space.

# Statement of need

The seaduck package is different from other ocean analytical tools (e.g. oceanspy \[@Almansi2019\]) in the sense that it accesses the data from a point's perspective. Users define the point's position using longitude, latitude, depth, and time, and the package determines which neighboring model grid points are needed to construct the continuous field around the point of interest. This is done using the efficient cKDtree object in the scipy.spatial module\[@Virtanen2020\]. Then, a fast space-time interpolation, which is compiled by numba\[@Lam2015\], returns the desired field value. As the points can be defined arbitrarily in the model domain, accessing discrete numerical output feels like getting values from a continuous field.

Users can define the shape of the interpolation kernel in both spatial and temporal dimensions. The kernel defines which neighboring points are used, and therefore how the continuous field is estimated. Aside from interpolation kernels, users can also calculate derivatives in all four dimensions with derivative kernels.

Large kernels are generally preferred if one wants to get a smooth, accurate result. However, many regions of interest in oceanography are close to land, and erroneous results can be generated if the interpolation kernel overlaps with the land. To handle this issue, seaduck's kernels are aware of the model land mask. Users can define a hierarchical sequence of kernels. If the first kernel overlaps with land, the second one is tried, and so on. The process is repeated until the kernel is small enough to avoid any land points, or all the kernels fail to avoid land and `np.nan` is returned. This function makes seaduck a handy and customizable tool for re-gridding datasets.

These functionalities are naturally considered from space-time points pre-specified by the user, namely from the Eulerian perspective (using the`seaduck.eulerian.Position` class).
The seaduck package can perform the same computations from the Lagrangian perspective, which follows points (also called particles) moving with the circulation (using the `seaduck.lagrangian.Particle` class).
Most Lagrangian particle packages compute particle trajectories numerically.
Instead, seaduck uses mass-conserving analytic formulae based on the assumption of a step-wise steady velocity field similar to that used by TRACMASS\[@tracmass\]. Just like the continuous fields in space, the particle positions can be defined as continuous functions of time.

The code to compute the Lagrangian particle trajectories is also compiled with numba\[@Lam2015\], which gives much better performance than interpreted Python code. Another specialty of this Lagrangian code is that it also supports datasets with complex face topology (e.g. ECCO\[@Forget2015\]).
