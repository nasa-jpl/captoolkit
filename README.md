![splash](splash.png)

# captoolkit - JPL Cryosphere Altimetry Processing Toolkit

Set of tools for processing and integrating satellite and airborne altimetry data.

## Credits

* [Fernando Paolo](https://science.jpl.nasa.gov/people/Serrano%20Paolo/) (paolofer@jpl.nasa.gov) - Main developer
* [Johan Nilsson](https://science.jpl.nasa.gov/people/Nilsson/) (johan.nilsson@jpl.nasa.gov) - Main developer
* [Alex Gardner](https://science.jpl.nasa.gov/people/AGardner/) (alex.s.gardner@jpl.nasa.gov) - Project PI

Jet Propulsion Laboratory, California Institute of Technology

## Installation

    git clone https://github.com/fspaolo/captoolkit.git
    cd captoolkit
    python setup.py install

## Example

Read ICESat-2 (ATL06) data files and extract some variable using 4 cores (from the command line):

    readatl06.py -n 4 *.h5 

See some [Jupyter Notebooks](notebooks/) for more examples.

## Notes

This package is constantly being updated, and new tools are being added as we finish testing them (many more will be included).

Currently, the individual programs work as standalone command-line utilities. We are working to convert them into a library, i.e., importable within your custom Python script.

There is no need to install the package to use the tools. You can simply do:

    python script.py -a arg1 -b arg2 /path/to/files/*.h5

## Utilities

* `readatl06.py` - Reads ICESat-2 ATL06 HDF5 and extract variables of interest
* `readglas.py` - Reads ICESat GLA12 Release 634 HDF5 and applies/removes corrections
* `readra.py` -  Reads ERS (REAPER) and applies/removes corrections
* `readra2.py` -  Reads Envisat and applies/removes corrections

* `lrmproc.py` - Processes CryoSat-2 LRM-mode for producing L2 data from ESA L1b product
* `sinproc.py` - Processes CryoSat-2 SIN-mode for producing L2 data from ESA L1b product
* `swathproc.py` - Processes CryoSat-2 (swath processor) for producing L2 data from ESA L1b product

* `slp2ibe.py` - Converts ERA-Interim Sea-level pressure to Inverse Barometer Effect (IBE)
* `ibecor.py` - Computes and applies the inverse barometer correction (IBE) to height data
* `tidecor.py` - Computes and applies tide corrections (using a 3rd-party software)
* `slopecor.py` - Corrects for slope-induced errors, using the 'direct' or 'relocation' method
* `scattcor.py` - Corrects radar altimetry height to correlation with waveform parameters

* `topofit.py` - Detrends data with respect to mean topography (linear or quadratic surface)
* `secfit.py` - Computes robust height changes using a surface-fit approach
* `atmfit.py` - Computes robust elevation changes from NASA's IceBridge ATM data
* `xover.py` - Computes crossover values at satellite orbit intersections

* `gcomb.py` - Manipulates rasters (addition, subtraction, multiplication, boolean operations, etc)
* `lscip.py` - Optimal interpolation of irregular data using least squares collocation (raster and point)
* `blockip.py` - Interpolates data using inverse distance weighting (raster and point data)
* `fcomp.py` - Computes statistics between spatial point datasets for satellite/airborne validation
* `topopar.py` - Estimates slope, aspect and curvature from given DEM, w/options for smoothing
* `clean.py` - Edits outliers of scattered point data, accounting for both temporal and spatial trends
* `mask.py` - Masks and selects scattered data using raster-mask, polygon or bounding box
* `gfilter.py` - Collection of multiple fast spatial and temporal filters, using JIT-compilation

* `setorbit.py` - Calculates unique identifiers for each multi-mission track (segments of data)
* `reftrack.py` - Identifies repeat tracks and calculates the reference ground tracks
* `septrack.py` - Separates orbit arcs into ascending and descending tracks
* `sepcampg.py` - Separates data points into campaigns (specified time intervals)

* `sortvars.py` - Sort (in place) all 1d variables in HDF5 file(s)
* `dummy.py` - Adds dummy variables as 1d arrays to HDF5 files(s)
* `txt2hdf.py` - Converts (very large) ASCII tables to HDF5 (1d arrays)
* `hdf2txt.py` - Converts HDF5 (1d-arrays) to ASCII (columns)

* `split.py` - Splits large 1d HDF5 file(s) into smaller ones
* `tile.py` - Tiles geographical data to reduce data volumes and allow parallelization
* `join.py` - Joins a set of geographical tiles (individual files)
* `merge.py` - Merges several HDF5 files into a single file or multiple larger files
* `mergetiles.py` - Merges tiles from different missions keeping the original tiling (grid)

