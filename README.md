![splash](splash.png)

# captoolkit - Cryosphere Altimetry Processing Toolkit

[![Language](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/fspaolo/captoolkit/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/captoolkit/badge/?version=latest)](https://captoolkit.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/104787010.svg)](https://zenodo.org/badge/latestdoi/104787010)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fspaolo/captoolkit/master)  
[![Binder](https://binder.pangeo.io/badge.svg)](https://binder.pangeo.io/v2/gh/fspaolo/captoolkit/master)


#### Set of tools for processing and integrating satellite and airborne (radar and laser) altimetry data.

## Project leads

* [Fernando Paolo](https://science.jpl.nasa.gov/people/Serrano%20Paolo/) (paolofer@jpl.nasa.gov)
* [Johan Nilsson](https://science.jpl.nasa.gov/people/Nilsson/) (johan.nilsson@jpl.nasa.gov)
* [Alex Gardner](https://science.jpl.nasa.gov/people/AGardner/) (alex.s.gardner@jpl.nasa.gov)

Jet Propulsion Laboratory, California Institute of Technology

Development of the codebase was funded by the NASA Cryospheric Sciences program and the NASA MEaSUReS ITS_LIVE project through award to Alex Gardner.

## Contributors

- Tyler Sutterley (tsutterl@uw.edu)

## Contribution

If you would like to contribute (your own code or modifications to existing ones) just create a [pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) or send us an email, we will gladly add you as a contributor to the project.

## Install

    git clone https://github.com/fspaolo/captoolkit.git
    cd captoolkit
    python setup.py install

## Example

Read ICESat-2 Land Ice Height product (ATL06) in parallel and extract some variables using 4 cores (from the command line):

    readatl06.py -n 4 *.h5

To see the input arguments of each program run:

    program.py -h

For more information check the header of each program.

## Notebooks

[Introduction to HDF5 data files](https://nbviewer.jupyter.org/github/fspaolo/captoolkit/blob/master/notebooks/intro-to-hdf5.ipynb)   
High-level overview of the HDF5 file structure and associated tools

[Reduction of ICESat-2 data files](https://nbviewer.jupyter.org/github/fspaolo/captoolkit/blob/master/notebooks/redu-is2-files.ipynb)  
Select (ATL06) files and variables of interest and write to a simpler structure

[Filtering and gridding elevation change data](https://nbviewer.jupyter.org/github/fspaolo/captoolkit/blob/master/notebooks/Gridding-rendered.ipynb)  
Interpolate and filter data to derive gridded products of elevation change

## Notes

This package is under heavy development, and new tools are being added as we finish testing them (many more utilities are coming).

Currently, the individual programs work as standalone command-line utilities or editable scripts. There is no need to install the package. You can simply run the python scripts as:

    python program.py -a arg1 -b arg2 /path/to/files/*.h5

## Tools

### Reading

* [`readgeo.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/readgeo.md) - Read Geosat and apply/remove corrections
* [`readers.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/readers.md) - Read ERS 1/2 (REAPER) and apply/remove corrections
* [`readra2.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/readra2.md) - Read Envisat and apply/remove corrections
* [`readgla12.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/readgla12.md) - Read ICESat GLA12 Release 634 HDF5 and apply/remove corrections
* * [`readgla06.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/readgla06.md) - Read ICESat GLA12 Release 634 HDF5 and apply/remove corrections
* [`readatl06.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/readatl06.md) - Read ICESat-2 ATL06 HDF5 and select specific variables

### Correcting

* [`corrapply.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/corrapply.md) - Apply a set of specified corrections to a set of variables
* [`corrslope.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/corrslope.md) - Correct slope-induced errors using 'direct' or 'relocation' method
* [`corrscatt.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/corrscatt.md) - Correct radar altimetry height to correlation with waveform parameters
* [`corrlaser.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/corrlaser.md) - Compute and apply corrections for ICESat Laser 2 and 3

### Filtering

* [`filtst.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/filtst.md) - Filter point-cloud data in space and time
* [`filtmask.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/filtmask.md) - Select scattered data using raster-mask, polygon or bounding box
* [`filtnan.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/filtnan.md) - Check for NaNs in a given 1D variable and remove the respective "rows"
* [`filttrack.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/filttrack.md) - Filter satellite tracks (segments) with along-track running window (**coming**)
* [`filttrackwf.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/filttrackwf.md) - Filter waveform tracks (segments) with along-track running window (**coming**)

### Differencing

* [`xing.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/xing.md) - Compute differences between two adjacent points (cal/val)
* [`xover.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/xover.md) - Compute crossover values at satellite orbit intersections

### Fitting

* [`fittopo.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/fittopo.md) - Detrend data with respect to modeled topography
* [`fitsec.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/fitsec.md) - Compute robust height changes using a surface-fit approach

### Interpolating

* [`interpgaus.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/interpgaus.md) - Interpolate irregular data using Gaussian Kernel
* [`interpmed.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/interpmed.md) - Interpolate irregular data using a Median Kernel
* [`interpkrig.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/interpkrig.md) - Interpolate irregular data using Kriging/Collocation

### Utilities

* [`split.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/split.md) - Split large 1D HDF5 file(s) into smaller ones
* [`merge.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/merge.md) - Merge several HDF5 files into a single or multiple file(s)
* [`mergetile.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/mergetile.md) - Merge tiles from different missions keeping the original grid
* [`tile.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/tile.md) - Tile geographical (point) data to allow parallelization
* [`join.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/join.md) - Join a set of geographical tiles (from individual files)
* [`joingrd.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/joingrd.md) - Join a set of geographical tiles (subgrids from individual files)
* [`stackgrd.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/stackgrd.md) - Stack a set of 2D grids into a 3D cube using time information
* [`sort.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/sort.md) - Sort (in place) all 1D variables in HDF5 file(s)
* [`dummy.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/dummy.md) - Add dummy variables as 1D arrays to HDF5 files(s)
* [`hdf2txt.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/hdf2txt.md) - Convert HDF5 (1D arrays) to ASCII tables (columns)
* [`txt2hdf.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/txt2hdf.md) - Convert (very large) ASCII tables to HDF5 (1D arrays)
* [`query.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/query.md) - Query entire data base (tens of thousands of HDF5 files) (**coming**)

### Gaussian Processes

* [`ointerp/ointerp.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/ointerp/ointerp.md) - Optimal Interpolation/Gaussian Processes
* [`ointerp/covx.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/ointerp/covx.md) - Calculate empirical spatial covariances from data
* [`ointerp/covt.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/ointerp/covt.md) - Calculate empirical temporal covariances from data
* [`ointerp/covfit.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/ointerp/covfit.md) - Fit analytical model to empirical covariances

### IBE

* [`ibe/corribe.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/ibe/corribe.md) - Compute and apply inverse barometer correction (IBE)
* [`ibe/slp2ibe.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/ibe/slp2ibe.md) - Convert ERA-Interim Sea-level pressure to IBE
* [`ibe/geteraint.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/ibe/geteraint.md) - Example python params to download ERA-Interim

### Tides

* [`tide/corrtide.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/tide/corrtide.md) - Compute and apply ocean and load tides corrections
* [`tide/calc_astrol_longitudes.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/tide/calc_astrol_longitudes.md) - Computes the basic astronomical mean longitudes
* [`tide/calc_delta_time.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/tide/calc_delta_time.md) - Calculates difference between universal and dynamic time
* [`tide/convert_xy_ll.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/tide/convert_xy_ll.md) - Convert lat/lon points to and from projected coordinates
* [`tide/infer_minor_corrections.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/tide/infer_minor_corrections.md) - Return corrections for 16 minor constituents
* [`tide/load_constituent.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/tide/load_constituent.md) - Loads parameters for a given tidal constituent
* [`tide/load_nodal_corrections.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/tide/load_nodal_corrections.md) - Load the nodal corrections for tidal constituents
* [`tide/predict_tide_drift.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/tide/predict_tide_drift.md) - Predict tidal elevations using harmonic constants
* [`tide/read_tide_model.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/tide/read_tide_model.md) - Extract tidal harmonic constants from OTIS tide models
* [`tide/read_netcdf_model.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/tide/read_netcdf_model.md) - Extract tidal harmonic constants from netcdf models
* [`tide/read_GOT_model.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/tide/read_GOT_model.md) - Extract tidal harmonic constants from GSFC GOT models

### 2D Fields

* [`gettopo.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/gettopo.md) - Estimate slope, aspect and curvature from given DEM
* [`getdem.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/getdem.md) - Regrid mean height field (DEM) from grid-1 onto grid-2
* [`getveloc.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/getveloc.md) - Combine best 2D mean field from different velocities
* [`vregrid.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/vregrid.md) - Regrid velocity field onto height field (**coming**)
* [`getmsl.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/getmsl.md) - Calculate and extend MSL field for the ice shelves
* [`mkmask.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/mkmask.md) - Compute ice shelf, basin and buffer raster masks (**coming**)

### 3D Fields

* [`cubefilt.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/cubefilt.md) - Filter slices (spatial) and individual time series (temporal)
* [`cubefilt2.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/cubefilt2.md) - Filter time series residuals w.r.t. a piece-wise poly fit
* [`cubexcal.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/cubexcal.md) - Cross-calibrate several data cubes with same dimensions
* [`cubeimau.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/cubeimau.md) - Filter and regrid IMAU Firn cube product
* [`cubegsfc.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/cubegsfc.md) - Filter and regrid GSFC Firn cube product
* [`cubegemb.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/cubegemb.md) - Filter and regrid JPL Firn and SMB cube products
* [`cubesmb.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/cubesmb.md) - Filter and regrid RACMO and ERA5 SMB cube products
* [`cubethick.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/cubethick.md) - Compute time-variable Freeboard, Draft, and Thickness
* [`cubediv.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/cubediv.md) - Compute time-variable Flux Divergence, and associated products
* [`cubemelt.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/cubemelt.md) - Compute time-variable basal melt rates and mass change
* [`cuberegrid.py`](https://github.com/fspaolo/captoolkit/blob/master/doc/source/user_guide/cuberegrid.md) - Remove spatial artifacts and regrid 3D fields

### Scripts

* `scripts/` - This folder contains supporting code (generic and specific) that we have used in our analyses. We provide these scripts **as is** in case you find them useful.

### Data

* `data/` - The data folder contains example data files for some of the tools. See respective headers.
