#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "2017-01-01/to/2019-01-31",  # update time interval
    "expver": "1",
    "grid": "0.75/0.75",
    "area": "-60/-180/-90/180",          # Subset an area (Antarctica) N/W/S/E
    "levtype": "sfc",
    "param": "151.128",                  # this is Mean Sea Level Pressure
    "time": "00:00:00/12:00:00",
    "step": "3/6/9/12",
    "stream": "oper",
    "type": "fc",
    "format": "netcdf",
    "target": "SLP_antarctica_3h_20170101_20190401.nc",
})


"""
server.retrieve({
    "class": "ea",
    "dataset": "era5_test",
    "date": "2016-01-01/to/2016-01-31", # Time period
    "expver": "12",
    "levtype": "sfc",
    "param": "sp",           # Parameters. Here we use Surface Pressure (sp). See the ECMWF parameter database, at http://apps.ecmwf.int/codes/grib/param-db
    "stream": "oper",
    "type": "an",
    "time": "00:00:00",
    "area": "75/-20/10/60",     # Subset or clip to an area, here to Europe. Specify as North/West/South/East in Geographic lat/long degrees. Southern latitudes and western longitudes must be given as negative numbers.
    "grid": "0.3/0.3",          # Regrid from the default Gaussian grid to a regular lat/lon with specified resolution. The first number is east-west resolution (longitude) and the second is north-south (latitude).
    "format": "netcdf",         # Convert the output file from the default GRIB format to NetCDF format.
    "target": "JUNK.nc",    # The output file name. Set this to whatever you like.
})
"""


# For global daily coverage: MSLP
"""
#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "1979-01-01/to/2017-01-31",
    "expver": "1",
    "grid": "0.75/0.75",
    "levtype": "sfc",
    "param": "151.128",
    "step": "0",
    "stream": "oper",
    "time": "00:00:00",
    "type": "an",
    "target": "output",
})
"""
