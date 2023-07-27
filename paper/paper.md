---
title:  'Seaduck: A python package for Eulerian and Lagrangian interpolation on ocean datasets'
tags:
  - Python
  - oceanography
  - interpolation
  - Lagarangian particle
authors:
  - name: Wenrui Jiang
    orcid: 0009-0000-7319-9469
    affiliation: 1
  - name: Thomas W. N. Haine
    orcid: 0000-0001-8231-2419
    affiliation: 1
  - name: Mattia Almansi
    orcid: 0000-0001-6849-3647
    affiliation: 2
    affiliations:
  - name: Department of Earth and Planetary Sciences, The Johns Hopkins University
    index: 1
  - name: B-Open Solutions s.r.l, Italy
    index: 2
date: 25 July 2023
bibliography: paper.bib
---

# Summary

Numerical simulations of the Earth's oceans are becoming more realistic and sophisticated. However, their complex layout and shear volume make it difficult for researchers to access and understand these data. Additionally, most ocean models,  mostly finite-volume models, compute and calculate spatially-integrated properties, such as grid-cell averaged temperature or wall-integrated mass flux. On the other hand, in-situ oceanographic observations are effectively collected at points in space and time. This fundamental difference makes the comparison between observations and results from numerical simulation difficult.

In this work, we present seaduck, a Python package that can perform both Eulerian and Lagrangian interpolation on generic ocean datasets with good performance and scalability. This package accesses numerical datasets from the perspective of space-time points. It automatically navigates complex layouts of datasets and transforms discrete information to continuous fields. The value and derivatives of those fields can be access at any points in the domain defined by the user. Similar to fixed and mobile observational platforms, the points can be either stationary (Eulerian) or advected by the flow (Lagrangian). 

# Statement of need

The seaduck package is different from other ocean analytical tools (e.g. oceanspy \[@Almansi2019\]) in the sense that it accesses the data from a point's perspective. Users define the points of interest using longitude, latitude, depth, and time, and the package then reads in relavent information from neighboring model grid points in discrete numerical models and constructs the continuous field around the points. Index lookup and space-time interpolation involved in this process is done efficiently with `scipy.spatial.cKDtree`[@Virtanen2020\] and numba\[@Lam2015\] compiled code , respectively. Since the points can be defined arbitrarily in the model domain, accessing discrete numerical output feels like getting values from a continuous field despite complex model layout.

The points can be stationary or be advected by a model vector field. Most Lagrangian particle packages (e.g. [@oceanparcel, @individualdisplacement]) that compute particle trajectories by solving initial value problems numerically. Seaduck, instead, uses a mass-conserving analytic scheme based on the assumption of a step-wise steady velocity field similar to that used by TRACMASS\[@tracmass\]. The advection code is largely numba compiled, and the total amount of computation is smaller than solving initial value problems. Furthermore, seaduck also allow users to access the analytical trajectory of the particle rather than interpolated ones. The Lagrangian particle functionality is built based on the above-mentioned interpolation utilities, thus, is automatically able to navigate complex topology of numerical models.

Highly customizable interpolation methods is available for both Eulerian (stationary) or Lagrangian points. Users can define all the properties of the kernel or the list of kernel used, including: (1) the shape of the interpolation kernel(s) in both spatial and temporal dimensions, which defines which neighboring points are used, and therefore how the continuous field is estimated. (2) The interpolation weight function, which allows users to calculate derivatives in all four dimensions apart from interpolation. (3) The hierarchical sequence of kernels, namely what is the next smaller kernel to use if some of the interpolation points are land-masked.

With the above suite of functionalities, seaduck is capable of accomplishing many common tasks in ocean dataset analysis including interpolation, regridding and Lagrangian particle simulation and some new ones including interpolation in Lagrangian label space and analyzing tracer budget from Lagrangian perspective. We also strive to make seaduck an accessible education tool by creating a very simple high-level interface intended for people with little programming background.

# Usage Examples

While some usage examples are presented here, many more can be found in the documentation for seaduck (https://macekuailv.github.io/seaduck/). The notebooks  of the following examples run on SciServer[@sciserver], an openly available cloud compute resource. A supplementary GitHub repository (https://github.com/MaceKuailv/seaduck_sciserver_notebook) holds all SciServer notebooks, which is being continuously maintained. 

![Fig.1 (a) Scatterplot with colors showing the sea surface height value near Kangerdlugssuaq Fjord defined in the model and interpolated by seaduck.\label{fig:onlyone}. (b) Streaklines of particle advected by stationary 2D slice of the LLC4320 simulation, colors denotes the current speed.](fig1.png)

## Interpolation / regridding

In this subsection, we are going to explore the interpolation/regridding functionality of the package. As an example, we used a realistic simulation of the Kangerdlugssuaq Fjord [@Fraser2018] as an example. This is an MITgcm simulation with very uneven grid spacings, i.e. grids close or in the fjord is much more densely placed than the rest. For the interpolation on sea surface height field, we use all the center grid points of the datasets as well as another 60,000 points in a rectangular region where the model grid points are sparsely places (between 66.5N to 67N, between 28.5W to 34.5 W, 600 in longitudinal direction and 100 in latitudinal direction). As shown in Fig. 1a. The interpolated field matches the background field very well, even when the interpolation is happening close to land ocean interface. 

## Global particle simulation on LLC4320

In this example, a stationary 2D slice of the state of the art LLC4320[@llc4320] model is used. LLC4320 is a global kilometer-scale model with complex topology. 150,000 particles were released randomly and evenly on the globe and are simulated for 30 days. This simulation takes about an hour to run on SciServer[@sciserver]. Fig. 1b is the particle trajectories produced in this simulation spanning several tiles of the domain looking from the North pole. The colors denote the current speed.
