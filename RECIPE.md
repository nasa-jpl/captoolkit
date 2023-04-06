# Ice Shelf Thickness and Basal Melt Rate Inversion 

Recipe to construct pan-Antarctic estimates of ice shelf
thickness changes and basal melt rates derived from
multiple satellite radar and laser altimetry measurements.

This recipe is specific to the dataset constructed for the paper
"*Widespread slowdown in thinning rates of West Antarctic Ice Shelves*,
Paolo et al. (2023), *The Cryosphere*", and it is provided "as is"
merely to illustrate the numerous steps in the inversion procedure
and to show examples of code (argument) signatures.

Paper: <URL>   
Code: https://github.com/nasa-jpl/captoolkit  
Data: https://its-live.jpl.nasa.gov  


## NOTES

- Many of the strategies were implemented to be run on a supercomputer
- The data are split into multiple smaller files for parallelization
- Many routines accept multiple input files to process in parallel
- See each routine's source code (header) for further information
- Examples of actual (random) runs are given to illustrate code usage
- Some scripts have been renamed in the released `captoolkit` repository
- Although we did not use ICESat data in the paper, we include it here


## TIPS

- After each processing step backup the generated files (`copydata.sh`)
- Rename original file names and extensions, e.g. .H5 -> .h5 (`rename.py`)

If getting "list too long", do the following operations as:   

    (rm) find source/ -name *.h5 -delete
    (rm) find source/ -name *.h5 | xargs -i rm {}
    (mv) find source/ -name *.h5 | xargs -i mv {} target/
    (cp) find source/ -name *.h5 | xargs -i cp {} target/
    (ls) find source/ -name *.h5 | xargs -i ls {} | wc -l
    (ls) find source/ -name *.h5 | xargs -i tail {}


## HDF5

Additional routines used for handling HDF5 files:

- `split.py` - Split large HDF5 file(s) into smaller ones
- `merge.py` - Merge several HDF5 files
- `tile.py` - Split geographical data into (overlapping) tiles
- `join.py` - Join tiles in space (individual files)
- `mergetile.py` - Merge tiles in time keeping the original tiling
- `joingrd.py` - Join gridded tiles in space (individual files)
- `mergegrd.py` - Merge different 2d grids into a single file
- `stack.py` - Stack 2d grids into a 3d grid (cube)
- `splitgrd.py` - Split 3D grid into multiple 2D grids
- `bintime.py` - Split data in time with (overlapping) time windows
- `query.py` - Query files and extract variables within search radius

Note that:

- most scripts run in parallel (with a `n_jobs=N` argument) 
- most scripts read and write in chunks from/to disk
- all scripts run from the command line with arguments


## READ

- ERS: Use time > 1992.25 (cut off at read time)
- ENV: Use time < 2010.75 (cut off at read time)
- ER1: Use "ocean mode" only for ERS-1. Too little data for ERS-2.
- CS2: Is read from Level 1B (and retracked in-house [1]),
  so not all geophysical params are available (e.g. tide-related vars)

[1]https://doi.org/10.5194/tc-10-2953-2016

Data are separated by mode and stored as HDF5, e.g.

    /path/to/ers1/read/*_OCN_*
    /path/to/ers1/read/*_ICE_*

Example run:

    python readers.py /path/to/ers1/RAW /path/to/ers1/read /path/to/masks/ANT_floatingice_240m.tif 3031 A 400 16 ocean PREFIX_ > readers1.log

    readra2.py /path/to/envisat/V3.0/RA2_GDR_2P /path/to/envisat/read /path/to/masks/ANT_floatingice_240m.tif 3031 A 400 16 > readra2.log

    python readgla.py /path/to/icesat/RAW/GLAH12.034 /path/to/icesat/read /path/to/masks/ANT_floatingice_240m.tif 3031 A 600 1 16 > readgla.log


## TRACKFILT

- Use `trackfilt_wf.py` for radar data with waveform params
- Rename extension of ICESat files: .H5 -> .h5 (`rename.py`)

Example run:

    python trackfilt_wf.py -f "/path/to/ers1/latest/*.h5" -v t_sec h_cor bs lew tes -a -n 16 &&

    python trackfilt.py -f "/path/to/icesat/latest/*.h5"  -v t_sec h_cor -a -n 16 &&


## LASERCOR

- Apply laser/campaign bias correction for ICESat (`lasercor.py`)


## SLOPECOR

- No need slope correction for ICESat and Cryosat-2

Example run:

    python slopecor.py '/path/to/ers1/latest/*.h5' -s /path/to/DEM/slope/bedmap2_surface_wgs84_2km_slope.tif -a /path/to/DEM/aspect/bedmap2_surface_wgs84_2km_aspect.tif --u /path/to/DEM/curve/bedmap2_surface_wgs84_2km_curve.tif --m RM -v lon lat h_cor range -l 1.0 -g A -n 16 &&


## TIMEFIX

- DEPRECATED: (temporary fix) This should be done at read time (`timefix.py`)


## IBECOR

- DO NOT APPLY, only compute values
- Compute using both `t_sec` and `t_sec_orig`

Example run:

    python ibecor.py "/path/to/envisat/latest/*.h5" -b /path/to/ibe/IBE_antarctica_3h_19900101_20171031.h5 -v lon lat t_sec h_cor -e 1970 1 1 0 0 0 -t 2001.5 2011.5 &&


## MERGE

Example run:

    python merge.py '/path/to/ers1/latest/*_OCN_*_A_*_IBE2.h5' -o /path/to/ers1/latest/ER1_OCN_READ_A_FILT_RM_TIMEFIX_IBE_IBE2.h5 -m 16 -z gzip -n 16 &&

    python merge.py '/path/to/cryosat2/latest/*_D_*_IBE2.h5' -o /path/to/cryosat2/latest/CS2_READ_D_FILT_RM_TIMEFIX_IBE_IBE2.h5 -m 16 -z gzip -n 16 && 


## TIDECOR

- DO NOT APPLY, only compute values
- This uses the Matlab code at https://github.com/fspaolo/tmdtoolbox
- There is also a python wrapper included in `captoolkit`
- Processing time ALL missions with 16 cores: ~6 hours

1. Edit header of `tidecor.m` (single script containing all missions)
2. Run multiple tests sequentially with `tidecor.sh`
3. Var `h_tide_sol1` is `h_tide + h_load + h_eq` (does not include `h_noneq`)

Example run:

    % Edit paths to HDF5s
    PATHS = {'/path/to/ers1/latest/ER1_OCN_READ_A_*',
             '/path/to/ers1/latest/ER1_OCN_READ_D_*',
             '/path/to/ers1/latest/ER1_ICE_READ_A_*',
             '/path/to/ers1/latest/ER1_ICE_READ_D_*',
             '/path/to/ers2/latest/ER2_ICE_READ_A_*',
             '/path/to/ers2/latest/ER2_ICE_READ_D_*',
             '/path/to/envisat/latest/ENV_READ_A_*',
             '/path/to/envisat/latest/ENV_READ_D_*',
             '/path/to/cryosat2/latest/CS2_READ_A_*',
             '/path/to/cryosat2/latest/CS2_READ_D_*',
             '/path/to/icesat/latest/IS1_READ_A_*',
             '/path/to/icesat/latest/IS1_READ_D_*'};

    % Run on the command line 
    /path/to/Applications/Matlab/bin/matlab -nodesktop < tidecor.m


## QUERY

- Verify tides and ibe

Example run:

    python query.py /path/to/envisat/latest/*_TIDE.h5 -o ENV_TIDE_QUERY.h5 -r 5 -v lon lat t_sec t_sec_orig t_year h_cor h_ibe h_ibe2 h_inv_bar h_tide h_load h_tide_sol1 h_tide_sol2 h_load_sol1 h_load_sol2 h_tide_eq h_tide_noneq orb_type &


## DUMMY

- Add asc/des flag to ICESat files

Example run:

    python dummy.py -f /path/to/icesat/latest/*_A_*_TIDE.h5 -v orb_type -l 0 -n 16
    python dummy.py -f /path/to/icesat/latest/*_D_*_TIDE.h5 -v orb_type -l 1 -n 16


## NANFILT

Example run:

    python nanfilt.py /path/to/ers1/latest/*_TIDE.h5 /path/to/ers2/latest/*_TIDE.h5 /path/to/envisat/latest/*_TIDE.h5 /path/to/cryosat2/latest/*_TIDE.h5 /path/to/icesat/latest/*_TIDE.h5 -v h_cor -n 16


## TILE

- First check grid bbox and tile boundaries with `plot_tilegrd.py`

Example run:

    python tile.py /path/to/ers1/latest/*_OCN_*_A_*_NONAN.h5 -b -2678407.5 2816632.5 -2154232.5 2259847.5 -d 100 -r 25 -v lon lat -j 3031 &

    python tile.py /path/to/ers1/latest/*_ICE_*_D_*_NONAN.h5 -b -2678407.5 2816632.5 -2154232.5 2259847.5 -d 100 -r 25 -v lon lat -j 3031 &


## MERGETILE 

Example run:

    python mergetile.py "/path/to/ers1/latest/*_OCN_*_A_*_tile*" -o /path/to/ers1/latest/ER1_OCN_READ_A_FILT_RM_TIMEFIX_IBE_IBE2_TIDE_NONAN.h5 -n 16 &&

    python mergetile.py "/path/to/ers1/latest/*_ICE_*_D_*_tile*" -o /path/to/ers1/latest/ER1_ICE_READ_D_FILT_RM_TIMEFIX_IBE_IBE2_TIDE_NONAN.h5 -n 16 &&


## UPLOAD

- Upload data to cluster: local server -> JPL clusters

Example run:

    rsync -av user@devon.jpl.nasa.gov:/path/to/cryosat2/latest/*_FILT_RM_TIMEFIX_IBE_IBE2_TIDE_NONAN_*_tile_* /cluster/path/to/cryosat2/latest/ 


## APPLYCOR

- Apply IBE, TIDE and LOAD
- Double check this correction has not been applied

Example run:

    python applycor.py /cluster/path/to/envisat/latest/*.h5 -v h_cor -c h_ibe h_tide h_load -n 16 &&


## TIMESPAN

- DEPRECIATED: (temporary fix) This should be done at read time 
- Filter out ICESat-1 campaigns: l1a, l1b, l2d, l2e, l2f 

Example run:

    python timespan.py /cluster/path/to/icesat/latest/*.h5


## LASERCOR

- DEPRECIATED: (temporary fix) This should be done at read time 
- Apply Laser-2/-3 correction according to Borsa et al. (2019)

Example run:

    python lasercor.py /cluster/path/to/icesat/latest/*_200880.h5


## QUERY

- Check the above corrections before applying topofit

Example run:

    python query.py /cluster/path/to/icesat/latest/*_LCOR.h5 -v lon lat t_year h_cor laser_bias
    rsync -av user@aurora.jpl.nasa.gov:/cluster/path/to/QUERY.h5 .
    python plot_query.py /path/to/QUERY.h5 -v lon lat t_year h_cor


## TOPOFIT

- Try filter by model order (e.g. m = 1 or 2, exclude m = 3)
  and see how much data gets removed. 

Example run:

    jobname = 'tf-env'
    files = '/cluster/path/envisat/latest/*_tile_*'
    cmd = 'python topofit.py <files> -d 1.0 1.0 -r 1.5 -i 5 -z 5 -k 1 -m 15 -q 2 -j 3031 -v lon lat t_year h_cor -b -2678407.5 2816632.5 -2154232.5 2259847.5 -t 2006 -n <ncpus>'
    

## QUERY

- Check how ICESat resolution came out
- Check filtering points by model order (m > 1)
- Check residuals were generated for all ice shelves
- Check residuals noise level

    python plot_query.py QUERY2.h5 -v lon lat t_year h_cor h_res h_mod m_deg e_res


## SCATCOR

- Check if all the tiles are present
- Query and test if bs reduces std

Example run:

    jobname = 'scat-env'
    files = "/cluster/path/to/envisat/latest/*_TOPO.h5"
    cmd = "python scattcor.py -f <files> -v lon lat h_res t_year -w bs lew tes -d 2 -r 5 -q 2 -p dif -b -2678407.5 2816632.5 -2154232.5 2259847.5 -a -n <ncpus>" 
    

## CLEANUP

- Cleanup data files every once in a while

Example run:

    python cleanup.py /cluster/cap/paolofer/cryosat2/floating/latest/*_SCAT.h5 -n 16 &&


## JOIN

- Join SCATGRD files for analysisi/testing
- These files are 1d arrays w/centroids (not 2d arrays)

Example run:

    python join.py /cluster/path/to/ers2/latest/*_ICE*_A_*_tile*_TOPO_SCATGRD.h5 -k tile -o ER2_ICE_A_SCATGRD_DIF.h5 &


## MERGETILE

- Merge OCN + ICE + ASC + DES (keeping tiles) for optimal interpolation

Example run:

    python mergetile.py /cluster/path/to/cryosat2/latest/*_SCAT.h5 -o /cluster/path/to/cryosat2/latest/CS2_AD.h5 -n 16 &&


## STFILTER

- This should have same -d and -r (variable) as ointerp
- Do no apply `stfilter.py`/`filtst.py` to icesat
- Set n_std in header

Example run:

    jobname = 'stf-cs2'
    files = '/cluster/path/to/cryosat2/latest/*_AD_*'
    cmd = 'python stfilter.py <files> -d 3 3 -v t_year lon lat h_res -b -2678407.5 2816632.5 -2154232.5 2259847.5 -n <ncpus>'
    

## NANFILT

Example run:

    python nanfilt.py /cluster/path/to/ers1/latest/*_STFILT.h5 /cluster/path/to/ers2/latest/*_STFILT.h5 /cluster/path/to/envisat/latest/*_STFILT.h5 /cluster/path/to/cryosat2/latest/*_STFILT.h5 /cluster/path/to/icesat/latest/*_AD_* -v h_res -n 16


## CLEANUP

Example run:

    python cleanup.py /cluster/path/to/icesat/latest/*_NONAN.h5


## BACKUP

- Make copy of processed files so far
- Query and verify the results so far 


## SECFIT

- This step is only to illustrate the use of `secfit.py` (skip to next)

    jobname = 'sf-is1'
    files = "/cluster/path/to/icesat/latest/*_AD_*_NONAN.h5"
    cmd = "python secfit.py <files> -m g -d 0.25 0.25 -r 0.3 0.3 -i 10 -z 10 -f 2006 -l 15 -q 2 -y 2 -s 1 -k 1 -u 0 -w 10 -j 3031 -v lon lat t_year h_res orbit None None None -p 1 -n <ncpus>"

    python join.py /cluster/cap/paolofer/icesat/floating/latest/*_sf.h5 -k tile -o ICE_AD_SECFIT_NEW2.h5


## BINTIME

- It must be done before calculating covariances

Example run:

    jobname = 'bin-env'
    files = "/cluster/path/to/envisat/latest/*_AD_*_NONAN.h5"
    cmd = "python bintime.py <files> -o /cluster/path/to/envisat/latest -t 2002 2011 -d 0.25 -w 0.416666667 -v t_year -n <ncpus>"
    

## OINTERP

- Fuse data to construct time series (3D fields)
- covx.py -> calculate spatial covariances
- covt.py -> calculate temporal covariances
- covfit.py -> model empirical covariances
- ointerp.py -> optimal interpolate

Example run:

    jobname = 'oi-cs2'
    files = "/cluster/path/to/cryosat2/latest/*_time_*"
    cmd = "python ointerp.py <files> -v lon lat h_res 0.35 -t t_year -d 3 3 -b -2678407.5 2816632.5 -2154232.5 2259847.5 -s _INTERP -n <ncpus>"


## JOINGRD

Example run:

    python joingrd.py /cluster/path/to/cryosat2/latest/*_INTERP -b -2678407.5 2816632.5 -2154232.5 2259847.5 -k tile -o /cluster/path/to/cryosat2/ointerp/CS2_OINTERP.h5 -n 16


## DOWNLOAD

- Download data to local machine before stacking

    rsync -av user@aurora.jpl.nasa.gov:/cluster/path/to/icesat/ointerp/IS1_OINTERP* /path/to/icesat/ointerp/ 


## STACKGRD

Example run:

    python stackgrd.py /path/to/cryosat2/ointerp/*.h5_time_* -v x y t_year -o /path/to/cryosat2/ointerp/CS2_CUBE.h5 &&


## CUBEFILT

- Filter slices (spatial) and individual time series (temporal)
- h_res -> h_res_filt (saved in the same HDF5)

Example run:

    python cubefilt.py /path/to/ers1/ointerp/*_CUBE.h5 /path/to/ers2/ointerp/*_CUBE.h5 /path/to/envisat/ointerp/*_CUBE.h5 /path/to/cryosat2/ointerp/*_CUBE.h5


## CUBEXCAL

- Cross-calibrate several cubes with same dimensions
- Multiple cubes into single cube (ER1_CUBE.h5, .., CS2_CUBE.h5 -> FULL_CUBE.h5)

Example run:

    python cubexcal.py /path/to/ers1/ointerp/*_CUBE.h5 /path/to/ers2/ointerp/*_CUBE.h5 /path/to/envisat/ointerp/*_CUBE.h5 /path/to/cryosat2/ointerp/*_CUBE.h5


## CUBEFILT2

- Filter time series residuals w.r.t. a piece-wise polynomial fit

Example run:

    python cubefilt2.py  # edit header


## GETDEM

- Compute 2D Mean height field from any of the single satellites,
  from all satellites (mean DEM), or use an external DEM (e.g. RIMA)


## VREGRID

- Regrid velocity fields

Three Options (codes):    

1. getveloc.py - combine best 2D mean field from Gardner et al. + Rignot et al.
2. vregrid1.py - regrid 2D summary field from Gardner et al. (already combined)
3. vregrid2.py - stack/regrid/combine time-variable velocities with mean 2D field

NOTE: Only use (3) vregrid2.py for time-variable velocity (as in this study)


## GETMSL
    
- Compute 2D MSL from Geoid, Armitage et al.'s MSL, GMDT, SLT 
- In future, include Armitage et al.'s MSL climatology, SLA, and SLT
- Double check for MDT potential shift


## CUBEGEMB

Compute ALL options:  

- cubeimau.py - FAC IMAU
- cubegsfc.py - FAC GSFC
- cubegemb.py - FAC and SMB GEMB
- cubesmb.py - SMB RACMO and ERA5

For FAC and SMB:

- Convert SMB units (m.i.eq./yr)
- Average at 5-M windows
- Regrid at 3 km


## CUBETHICK

- Compute time-variable (3D) Freeboard, Draft, and Thickness 
- Alternatively, look at cubedem.py (2D DEM + anomalies -> 3D series)

Output products (all corrected for FAC and SLT):

- Height time series (3D)
- Freeboard time series (3D) 
- Draft time series (3D) 
- Thickness time series (3D)  


## CUBEDIV

- Compute time-variable (3D) Flux Divergence, and associated products
- The calculation of divergence per slice could be parallelized (it's too slow)
- Save the results of `cubediv.py` into a smaller `FULL_CUBE_REDUCED.h5`,
  and use this smaller file from now on for efficiency.


## CUBEMELT

- Compute time-variable (3D) melt rates
- Compute time-variable (3D) net mass change
- Fill in the polehole
- Filter/smooth some fields

NOTE: Determine the sign of melt (can be either positive or negative). 


## CUBEREGRID

- Remove velocity boundary artefacts
- Regrid 3000 m fields to 960 m (for MEASURES)
- Recompute Mean Melt Rate field at 960 m
- Generate 960-m ice shelf masks


## MKMASK

- Compute ice shelf and basin raster masks
- Compute buffers for floating and grounded ice

NOTE: This step is done only once, and it's specific to cube resolution


## CUBEGROUND

- Incorporate Nilsson et al. and Schröder et al. grounded ice cubes
- Resample grounded cubes (t,y,x) onto floating cube (y,x,t)


## CUBEERROR

- Estimate and propagate uncertainties for
    * h (dh_xcal)
    * H
    * dH/dt
    * div
    * melt


## VERSIONS

Many versions of the whole processing have been run, including 
using static vs variable velocities, simple MSL and SLT extrapolation, 
various FAC and SMB models, and different filtering techniques. 
A thorough analysis was performed to select the best solution.
