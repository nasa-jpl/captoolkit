#!/usr/bin/env python

import os
from glob import glob
from distutils.core import setup

setup(
    name='captoolbox',
    version='0.1.0',
    license='Apache',
    author=['Johan Nilsson', 'Fernando Paolo'],
    author_email=['johan.nilsson@jpl.nasa.gov', 'paolofer@jpl.nasa.gov'],
    url='https://github.com/fspaolo/captoolbox',
    download_url='https://github.com/fspaolo/captoolbox.git',
    description='JPL Cryosphere Altimetry Processing Toolbox',
    long_description=open('README.md').read(),
    packages=['captoolbox', 'captoolbox.lib'],
    #NOTE: If Anaconda Python, this will write scripts
    # to ~/anaconda2/bin, else to /usr/local/bin 
    scripts=glob(os.path.join('captoolbox', '*.py')),
)
