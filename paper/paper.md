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
link-bibliography: false
---

# Summary

Numerical simulations of the Earth's oceans are becoming more realistic and sophisticated. Their complex layout and sheer volume make it difficult for researchers to access and understand these data, however. Additionally, most ocean models, mostly finite-volume models, compute and calculate spatially-integrated properties, such as grid-cell averaged temperature or wall-integrated mass flux. On the other hand, in-situ oceanographic observations are effectively collected at points in space and time. This fundamental difference makes the comparison between observations and results from numerical simulation difficult.

In this work, we present seaduck, a Python package that can perform both Eulerian interpolation and Lagrangian particle simulation on generic ocean datasets with good performance and scalability. This package accesses numerical datasets from the perspective of space-time points. It automatically navigates complex dataset layouts (grid topologies) and transforms discrete information into continuous fields. The values and derivatives of those fields can be accessed at any point in the 3+1 dimensional space-time domain defined by the user. For example, users can compute oceanographic properties as Eulerian timeseries (at fixed points in space), like from moored oceanographic current meters, or as Lagrangian timeseries (drifting with the current), like from oceanographic floats


# Statement of need

The seaduck package is different from other ocean analytical tools (e.g., oceanspy [@Almansi2019]) because it accesses the circulation model data from the perspective of an arbitrary space-time point. Users define the points of interest using longitude, latitude, depth, and time. The package then reads necessary information from nearby model grid points and constructs the continuous (scalar or vector) field around the points. The index lookup and space-time interpolation involved in this process is done efficiently with `scipy.spatial.cKDtree` [@Virtanen2020] and numba [@Lam2015] compiled code, respectively. As the points can be defined arbitrarily in the model domain, accessing discrete numerical output feels to the user like retrieving values from a continuous field, despite the complex model grid.

The points can be stationary (fixed in space, or Eulerian) or be advected by a vector velocity field (Lagrangian). Most Lagrangian particle packages (e.g., [@oceanparcel, @individualdisplacement]) compute particle trajectories by solving the initial value problem numerically. Instead, seaduck uses efficient, accurate, mass-conserving analytic formulae, which assumes a step-wise steady velocity field similar to that used by TRACMASS [@tracmass]. The Lagrangian advection code is largely numba compiled, and the total amount of computation is less than solving the problem numerically. The Lagrangian particle functionality is based on the above-mentioned interpolation utilities, thus, it automatically navigates the complex topology of numerical ocean models.

Seaduck's interpolation functionality is similar to other interpolation tools (e.g., scipy[@Virtanen2020] and xESMF[xesmf]).
Seaduck only supports interpolation using neighboring grid points, but users control the properties of a hierarchy of interpolation kernels; which is an advantage. These properties include: (1) The space-time shape of the interpolation kernel(s), which defines which neighboring points are used, and therefore how the continuous field is estimated. (2) The kernel weight function, which allows users to calculate generic linear operations on the data beyond simple interpolation, such as differentiation and smoothing. (3) The hierarchy of kernels near land-masked points.

With the above functionality, seaduck can accomplish many common tasks in ocean model data analysis, including interpolation, regridding, and Lagrangian particle simulation. Less common tasks are also possible, such as interpolation in Lagrangian label space, and analysis of tracer budgets along Lagrangian trajectories. We also strive to make seaduck an accessible education tool by creating a very simple high-level default interface, which is intended for people with little programming background, and for people who want to quickly try the tool.

# Usage Examples

While some usage examples are presented here, many more can be found in the documentation for seaduck (https://macekuailv.github.io/seaduck/). The notebooks of the following examples run on SciServer [@sciserver], an openly available cloud compute resource for scientific data analysis. A supplementary GitHub repository (https://github.com/MaceKuailv/seaduck_sciserver_notebook) holds all SciServer notebooks, and is being continuously maintained.

## Interpolation / regridding

As an example of seaduck's interpolation/regridding functionality,  consider a realistic simulation of the Kangerdlugssuaq Fjord, which is  in east Greenland [@Fraser2018]. The simulation uses the MITgcm [@mitgcm] with uneven grid spacing such that grid cells within the fjord are much more densely packed than elsewhere. The goal is to interpolate, and  hence regrid, the sea surface height field, , to a uniform grid spacing in the southern part of the domain. In  Figure. 1, the coherent patch between 66.5 N and 67 N is a very dense  scatter plot of the interpolated  field.  The close agreement between the interpolated and output value  can be clearly seen in Figure. 1. The interpolated field also remains  smooth near strong gradients and land boundaries.

![Scatterplot with colors showing the sea surface height value near Kangerdlugssuaq Fjord defined in the model and interpolated by seaduck.\label{fig:onlyone}](https://github.com/MaceKuailv/seaduck_sciserver_notebook/blob/master/stable_images/Fjord_29_0.png?raw=true)

## Global particle simulation on LLC4320

In this example, a stationary, surface slice of the LLC4320 [@llc4320] simulation is used. LLC4320 is a kilometer-scale model of the global ocean circulation with complex topology. 150,000 Lagrangian particles are released randomly and evenly on the globe, and seaduck computes their trajectories for 30 days. Figure. 2 shows the particle trajectories for the northern hemisphere, which contains around 10$^8$ velocity points. The colors denote the current speed. This simulation takes about an hour to run on SciServer [@sciserver].

![Particle streaklines of particle advected by stationary 2D slice of the LLC4320 simulation. Colors denote the current speed.](https://github.com/MaceKuailv/seaduck_sciserver_notebook/blob/master/stable_images/LLC4320_29_2.png?raw=true)

# Acknowledgments

The authors thank Erik van Sebille and Alex Szlay for enlightening discussions during the development of the package.

# References
