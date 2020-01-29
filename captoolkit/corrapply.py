#!/usr/bin/env python
"""
Apply a set of specified of corrections to a set of variables.

Add or Substract 1D variables to/from other 1D variables
in the same HDF5 file(s).

Example:
    python corrapply.py -h

Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit

    Fernando Paolo (paolofer@jpl.nasa.gov)
    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)

    Jet Propulsion Laboratory, California Institute of Technology

"""
import os
import h5py
import warnings
import argparse

# from glob import glob

warnings.filterwarnings("ignore")


def get_args():
    msg = "Apply correction(s) to specific variable(s)"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument(
        "file", type=str, nargs="+", help="HDF5 file(s) to read"
    )
    parser.add_argument(
        "-v",
        metavar=("var"),
        dest="vnames",
        type=str,
        nargs="+",
        help=("name of variables in the HDF5 to correct"),
    )
    parser.add_argument(
        "-c",
        metavar=("cor"),
        dest="cnames",
        type=str,
        nargs="+",
        help=("name of corrections in the HDF5 to apply"),
    )
    parser.add_argument(
        "-u",
        dest="add",
        action="store_true",
        help=("to unapply (add) corrections instead"),
        default=False,
    )
    parser.add_argument(
        "-n",
        metavar=("n_jobs"),
        dest="njobs",
        type=int,
        nargs=1,
        help="for parallel processing of multiple files",
        default=[1],
    )
    return parser.parse_args()


def rename_file(fname, suffix="_COR"):
    path, ext = os.path.splitext(fname)
    os.rename(fname, path + suffix + ext)


def main(fname, vnames, cnames, add):
    with h5py.File(fname, "a") as f:
        for cor in cnames:
            for var in vnames:
                if add:
                    f[var][:] += f[cor][:]
                else:
                    f[var][:] -= f[cor][:]

    rename_file(fname)


args = get_args()
files = args.file[:]
vnames = args.vnames[:]
cnames = args.cnames[:]
njobs = args.njobs[0]
add = args.add


if njobs == 1:
    print("running sequential code ...")
    [main(f, vnames, cnames, add) for f in files]

else:
    print(("running parallel code (%d jobs) ..." % njobs))
    from joblib import Parallel, delayed

    Parallel(n_jobs=njobs, verbose=5)(
        delayed(main)(f, vnames, cnames, add) for f in files
    )

print("done.")
