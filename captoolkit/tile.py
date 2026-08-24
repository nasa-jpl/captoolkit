#!/usr/bin/env python
"""
Program for tiling geographical point data into spatially defined,
optionally overlapping tiles.

Input data are provided in HDF5 format with longitude and latitude
coordinates. Tiles are generated in a user-specified projected coordinate
system (EPSG) using a defined tile size and optional overlap.

If a bounding box is provided with the -b option, the supplied projected
extent is used directly to generate the tiles, avoiding an initial read and
transformation of the full coordinate dataset. The input file is then read
once in chunks, coordinates are transformed once per chunk, and observations
are distributed to the corresponding tiles.

If no bounding box is provided, the full coordinate extent is first
determined from the input data in chunks before generating the tiles.

captoolkit - JPL Cryosphere Altimetry Processing Toolkit

Johan Nilsson (johan.n.nilsson@geo.uu.se)
Fernando Paolo (paolofer@jpl.nasa.gov)
Alex Gardner (alex.s.gardner@jpl.nasa.gov)

Jet Propulsion Laboratory, California Institute of Technology
Department of Earth Sciences, Uppsala University
"""

import os
import warnings
warnings.filterwarnings("ignore")
import pyproj
import argparse
import tables as tb
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Split geographical data into (overlapping) tiles')

    parser.add_argument('file', type=str, help='input file (HDF5 format)')

    parser.add_argument('-b', dest='bboxlim', type=float, nargs=4,
                        help='bounding box (xmin xmax ymin ymax) in projected meters, optional')

    parser.add_argument('-d', dest='dxy', type=float, required=True,
                        help='tile size (km)')

    parser.add_argument('-r', dest='dr', type=float, default=0,
                        help='buffer/overlap size (km)')

    parser.add_argument('-v', dest='vnames', type=str, nargs=2,
                        default=['lon', 'lat'],
                        help='names of longitude and latitude variables')

    parser.add_argument('-j', dest='proj', type=str, default='3031',
                        help='EPSG projection number')

    parser.add_argument('-n', dest='njobs', type=int, default=1,
                        help='number of parallel jobs (single-pass processing uses one job)')

    parser.add_argument('--chunk', dest='chunk_size', type=int, default=100000,
                        help='chunk size for out-of-core processing (number of rows per read)')

    return parser.parse_args()


def get_bbox(ifile, vnames, proj, chunk_size):
    xvar, yvar = vnames

    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{proj}", always_xy=True)

    with tb.open_file(ifile) as fi:
        xnode = fi.get_node('/', xvar)
        ynode = fi.get_node('/', yvar)

        nrows = len(xnode)

        for i in range(0, nrows, chunk_size):
            j = min(i + chunk_size, nrows)

            lon = xnode[i:j]
            lat = ynode[i:j]

            x, y = transformer.transform(lon, lat)

            valid = np.isfinite(x) & np.isfinite(y)

            if not np.any(valid):
                continue

            xmin = min(xmin, np.min(x[valid]))
            xmax = max(xmax, np.max(x[valid]))
            ymin = min(ymin, np.min(y[valid]))
            ymax = max(ymax, np.max(y[valid]))

    return xmin, xmax, ymin, ymax


def get_bboxs(bbox, dxy):
    xmin, xmax, ymin, ymax = bbox
    dxy = float(dxy)

    x_edges = np.arange(xmin, xmax, dxy)
    y_edges = np.arange(ymin, ymax, dxy)

    if x_edges.size == 0 or x_edges[-1] < xmax:
        x_edges = np.append(x_edges, xmax)

    if y_edges.size == 0 or y_edges[-1] < ymax:
        y_edges = np.append(y_edges, ymax)

    bboxs = [(w, e, s, n) for w, e in zip(x_edges[:-1], x_edges[1:])
                         for s, n in zip(y_edges[:-1], y_edges[1:])]

    return bboxs


def get_tiles(ifile, bboxs, vnames, buff=1000, proj='3031', chunks=100000):
    xvar, yvar = vnames

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{proj}", always_xy=True)

    xmin_all = min(b[0] for b in bboxs)
    xmax_all = max(b[1] for b in bboxs)
    ymin_all = min(b[2] for b in bboxs)
    ymax_all = max(b[3] for b in bboxs)

    outputs = [None] * len(bboxs)
    npts = np.zeros(len(bboxs), dtype=np.int64)

    with tb.open_file(ifile) as fi:
        vars_in_file = [fi.get_node('/', v.name) for v in fi.list_nodes('/')]
        lon_node = fi.get_node('/', xvar)
        lat_node = fi.get_node('/', yvar)

        nrows = len(lon_node)

        for i in range(0, nrows, chunks):
            j = min(i + chunks, nrows)

            print(f'Reading {i:,} - {j:,} of {nrows:,}')

            lon = lon_node[i:j]
            lat = lat_node[i:j]

            x, y = transformer.transform(lon, lat)

            mask_all = ((x >= xmin_all - buff) & (x <= xmax_all + buff) &
                        (y >= ymin_all - buff) & (y <= ymax_all + buff))

            if not np.any(mask_all):
                continue

            chunk_data = []

            for v in vars_in_file:
                if v.name == xvar:
                    data = lon
                elif v.name == yvar:
                    data = lat
                else:
                    data = v[i:j]

                chunk_data.append(data)

            for n, bbox in enumerate(bboxs):
                xmin, xmax, ymin, ymax = bbox

                mask = (mask_all &
                        (x >= xmin - buff) & (x <= xmax + buff) &
                        (y >= ymin - buff) & (y <= ymax + buff))

                if not np.any(mask):
                    continue

                if outputs[n] is None:
                    suffix = f"_bbox_{int(xmin)}_{int(xmax)}_{int(ymin)}_{int(ymax)}_buff_{buff/1000:.1f}_epsg_{proj}_tile_{n+1:03d}"
                    path, ext = os.path.splitext(ifile)
                    ofile = path + suffix + ext

                    fo = tb.open_file(ofile, 'w')
                    out_vars = [fo.create_earray('/', v.name, v.atom, shape=(0,))
                                for v in vars_in_file]

                    outputs[n] = (fo, out_vars)

                fo, out_vars = outputs[n]

                for out_var, data in zip(out_vars, chunk_data):
                    out_var.append(data[mask])

                npts[n] += np.count_nonzero(mask)

    for n, output in enumerate(outputs):
        if output is not None:
            fo, out_vars = output
            fo.flush()
            fo.close()
            print(f'Tile {n+1:03d}: {npts[n]:,} points saved.')
        else:
            print(f'Tile {n+1:03d}: No points in region.')


def main():
    args = get_args()

    ifile = args.file
    vnames = args.vnames
    bbox_lim = args.bboxlim
    dxy = args.dxy * 1000
    dr = args.dr * 1000
    proj = args.proj
    njobs = args.njobs
    chunk_size = args.chunk_size

    if njobs > 1:
        print("Single-pass processing uses one job to avoid rereading the input file.")

    if bbox_lim is not None:
        bbox = bbox_lim
    else:
        print("Determining data bounding box...")
        bbox = get_bbox(ifile, vnames, proj, chunk_size)

    print("Generating bounding boxes...")
    bboxs = get_bboxs(bbox, dxy)

    print(f"Number of tiles: {len(bboxs)}")
    print("Processing input file...")

    get_tiles(ifile, bboxs, vnames, buff=dr, proj=proj, chunks=chunk_size)


if __name__ == "__main__":
    main()