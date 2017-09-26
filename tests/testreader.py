import os
import sys
import h5py
import numpy as np
import pandas as pd


infile = 'cryosat_floatingice.txt'
outfile = os.path.splitext(infile)[0] + '.h5'
chunksize= 1000000
sep = ' '
names = [11, 22, 33, 44, 55, 66, 77, 88, 99]
names = [str(n) for n in names]  # must be strings


if 1:

    print 'determining file size...'
    num_rows = sum(1 for line in open(infile))  # scan file and count rows
    num_cols = len(open(infile).readline().split(sep))  # scan first row
    print 'file size is %d x %d' % (num_rows, num_cols)

    with h5py.File(outfile, 'w') as f:

        # Initialize empty datasets to hold the outputs
        dset = {name: f.create_dataset(name, shape=(num_rows,), dtype='float64') \
                for name in names}

        reader = pd.read_table(infile, sep=sep, names=names, chunksize=chunksize)
        nrow1, rnow2 = 0, 0

        for chunk in reader:

            nrow2 += chunk.shape[0]

            # Write the next chunk
            for name, d in zip(names, chunk.values.T):
                dset[name][nrow1:nrow2] = d

            # Increment the row count
            nrow1 += chunk.shape[0]

            print 'number of lines read:', nrow2, '...'

    print 'output ->', outfile

else:

    with h5py.File(outfile, 'w') as f:

        # Read the first chunk to get the column structure
        reader = pd.read_table(infile, sep=sep, names=names, chunksize=chunksize)
        chunk = reader.get_chunk(chunksize)
        nrows = chunk.shape[0]

        # Initialize resizable datasets to hold the outputs
        dset = {name: f.create_dataset(name, data=d, maxshape=(None,)) \
                for name,d in zip(names, chunk.values.T)}

        for chunk in reader:

            # Resize the datasets to accommodate the next chunk of rows
            [dset[name].resize(nrows + chunk.shape[0], axis=0) for name in names]

            # Write the next chunk
            for name,d in zip(names, chunk.values.T):
                dset[name][nrows:] = d

            # Increment the row count
            nrows += chunk.shape[0]

            print 'number of lines read:', nrows, '...'

    print 'output ->', outfile
