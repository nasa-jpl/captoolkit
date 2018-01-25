#!/usr/bin/env python

import os
from glob import glob
from distutils.core import setup

setup(
    name='captoolbox',
    version='0.1.0',
    license='Apache',
    author='Johan Nilsson and Fernando Paolo',
    author_email='johan.nilsson@jpl.nasa.gov, paolofer@jpl.nasa.gov',
    url='https://github.com/fspaolo/captoolbox',
    download_url='https://github.com/fspaolo/captoolbox.git',
    description='JPL Cryosphere Altimetry Processing Toolbox',
    long_description=open('README.md').read(),
    packages=['captoolbox', 'captoolbox.lib'],
    #NOTE: If Anaconda Python, the above will write executable scripts
    # to ~/anaconda2/bin, otherwise the default is /usr/local/bin 
    scripts=glob(os.path.join('captoolbox', '*.py')),
)
