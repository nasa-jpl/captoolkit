![splash](splash.png)

# captoolkit - JPL Cryosphere Altimetry Processing Toolkit

Set of tools for processing and integrating satellite and airborne altimetry data.

## Credits

* [Fernando Paolo](https://science.jpl.nasa.gov/people/Serrano%20Paolo/) (paolofer@jpl.nasa.gov) - Main developer
* [Johan Nilsson](https://science.jpl.nasa.gov/people/Nilsson/) (johan.nilsson@jpl.nasa.gov) - Main developer
* [Alex Gardner](https://science.jpl.nasa.gov/people/AGardner/) (alex.s.gardner@jpl.nasa.gov) - Project PI

Jet Propulsion Laboratory, California Institute of Technology

## Install

    git clone https://github.com/fspaolo/captoolkit.git
    cd captoolkit
    python setup.py install

## Example

Read ICESat-2 (ATL06) data files and extract some variable using 4 cores (from the command line):

    readatl06.py -n 4 *.h5 

To see the imput arguments of each program do:

    program.py -h

See some [Jupyter Notebooks](notebooks/) for more examples.

## Notes

This package is constantly being updated, and new tools are being added as we finish testing them (more utilities coming).

Currently, the individual programs work as standalone command-line utilities. There is no need to actually install the package. You can simply run the programs as:

    python script.py -a arg1 -b arg2 /path/to/files/*.h5

## Programs 

### Reading

* `readgeo.py`
* `readers.py`
* `readenvi.py`
* `readra2.py`
* `readgla12.py`
* `readatl06.py` - Reads ICESat-2 ATL06 HDF5 and select variables

### Correcting

* `corrapply.py`
* `corrslope.py`
* `corrscatt.py`
* `corrlaser.py`

### Filtering

* `filtmask.py`
* `filtnan.py`

### Differencing

* `xing.py`
* `xover.py`

### Fitting

* `fittopo.py`
* `fitsec.py`

### Interpolating

* `interpgaus.py`
* `interpkrig.py`

### Utilities

* `gettopo.py`
* `split.py`
* `merge.py`
* `mergetile.py`
* `tile.py`
* `join.py`
* `joingrd.py`
* `sort.py`
* `dummy.py`
* `hdf2txt.py`
* `txt2hdf.py`
