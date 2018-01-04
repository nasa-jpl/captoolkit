
Generate and apply the inverse barometer correction (IBE) to height data.

USAGE

Convert ERA-Interim mean sea-level pressure [Pa] to inverse barometer cor [m]:

    python slp2ibe.py -a -b file.nc

Apply the IB correction to an ASCII file with x,y,t in columns 0,1,2,
and specify the time units:

    python ibecor.py -a -b file.txt


NOTES

- For ERA-Interim the point interval on the native Gaussian grid is about
  0.75 degrees (keep the native resolution grid)

- On sufficiently long time scales and away from coastal effects, the ocean’s
  isostatic response is ~1 cm depression of sea level for a 1 hecto-Pascal (hPa)
  or milibar (mbar) increase in P_air (Gill, 1982; Ponte and others, 1991;
  Ponte, 1993).

- The frequency band 0.03 < w <0.5 cpd (T=2-33 days) (a.k.a. the "weather band")
  contains most of the variance in P_air. At higher frequencies, tides and
  measurement noise dominate h, and at lower frequencies, seasonal and
  climatological changes in the ice thickness and the underlying ocean state
  dominate the variability. 

- The IBE correction has generaly large spatial scales.

- There can be significant trends in P_air on time scales of 1-3 yr.

- ERA-Interim MSL pressure has time units: hours since 1900-01-01 00:00:0.0

- For MSLP we want forecast values only (without including the analyses steps)
  every 3h, so select: Times 0 and 12, Steps 3/6/9/12 

- To apply the IBE correction:

    h_cor = h - h_ibe


INSTRUCTIONS TO DOWNLOAD ERA-INTERIM DATA

(If you already have an account and the python library, skip to step 7)

1) Self register at:

    https://apps.ecmwf.int/registration/
    
2) Login:

    https://apps.ecmwf.int/auth/login/

3) Retrieve your key at:

    https://api.ecmwf.int/v1/key/

4) Copy the information in this page:

    {
        "url"   : "https://api.ecmwf.int/v1",
        "key"   : "XXXXXXXXXXXXXXXXXXXXXX",
        "email" : "john.smith@example.com"
    }

    and paste it to (Unix):

    $HOME/.ecmwfapirc

5) Accept the terms and conditions at:

    http://apps.ecmwf.int/datasets/licences/general

6) Install Python client library:

    sudo pip install https://software.ecmwf.int/wiki/download/attachments/56664858/ecmwf-api-client-python.tgz

7) Generate download script (geteraint.py):

    #!/usr/bin/env python
    from ecmwfapi import ECMWFDataServer
    server = ECMWFDataServer()
    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "date": "1990-01-01/to/2017-10-31",  # update time interval
        "expver": "1",
        "grid": "0.75/0.75",
        "area": "-60/-180/-90/180",          # subset an area (Antarctica) N/W/S/E
        "levtype": "sfc",
        "param": "151.128",                  # this is Mean Sea Level Pressure
        "step": "3/6/9/12",
        "stream": "oper",
        "time": "00:00:00/12:00:00",
        "type": "fc",
        "format": "netcdf",
        "target": "SLP_antarctica_3h_19900101_20171031.nc",
    })

8) Submit request:

    python geteraint.py

