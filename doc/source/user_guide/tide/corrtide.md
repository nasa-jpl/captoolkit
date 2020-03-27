corrtide.py
===========

- Calculates tidal elevations for correcting altimetry data
- Can use OTIS format tidal solutions provided by Ohio State University and ESR
- Can use Global Tide Model (GOT) solutions provided by Richard Ray at GSFC

#### Calling Sequence
```bash
python corrtide.py '/path/to/data/*.h5' \
    -D /path/to/tidedirectory -T CATS2008 \
    -v lon lat t_sec h_cor -a
```

#### Inputs
1. input HDF5 or ascii file

#### Command Line Options
- `-D X`, `--directory=X`: Working data directory for tide models
- `-T X`, `--tide=X`: Tide model to use in correction
    * CATS0201
    * CATS2008
    * CATS2008_load
    * TPXO9-atlas
    * TPXO9.1
    * TPXO8-atlas
    * TPXO7.2
    * TPXO7.2_load
    * AODTM-5
    * AOTIM-5
    * AOTIM-5-2018
    * GOT4.7
    * GOT4.7_load
    * GOT4.8
    * GOT4.8_load
    * GOT4.10
    * GOT4.10_load
- `-v X`, `--variable=X`: variable names of lon/lat/time/height in HDF5 file
- `-c X`, `--cols=X`: column positions of lon/lat/time/height in ASCII file
- `-e X`: reference epoch of input time in secs
- `-a`: apply tide correction instead of saving to separate file

#### Output options:
1. Applies tide correction and saves the cor as additional variable.
2. Generates external file with correction for each point (x,y,t,cor).

#### Dependencies:
- `calc_astrol_longitudes.py`: computes the basic astronomical mean longitudes
- `calc_delta_time.py`: calculates difference between universal and dynamic time
- `convert_xy_ll.py`: convert lat/lon points to and from projected coordinates
- `infer_minor_corrections.py`: return corrections for 16 minor constituents
- `load_constituent.py`: loads parameters for a given tidal constituent
- `load_nodal_corrections.py`: load the nodal corrections for tidal constituents
- `predict_tide_drift.py`: predict tidal elevations using harmonic constants
- `read_tide_model.py`: extract tidal harmonic constants from OTIS tide models
- `read_netcdf_model.py`: extract tidal harmonic constants from netcdf models
- `read_GOT_model.py`: extract tidal harmonic constants from GSFC GOT models
