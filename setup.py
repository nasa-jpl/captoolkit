#!/usr/bin/env python

import os
from glob import glob
from distutils.core import setup

setup(
    name='captoolkit',
    version='0.1.0',
    license='Apache',
    author='Fernando Paolo and Johan Nilsson',
    author_email='paolofer@jpl.nasa.gov, johan.nilsson@jpl.nasa.gov',
    url='https://github.com/fspaolo/captoolkit',
    download_url='https://github.com/fspaolo/captoolkit.git',
    description='JPL Cryosphere Altimetry Processing Toolkit',
    long_description=open('README.md').read(),
    packages=['captoolkit', 'captoolkit.lib'],
    #NOTE: If Anaconda Python, the above will write executable scripts
    # to ~/anaconda2/bin, otherwise the default is /usr/local/bin 
    scripts=glob(os.path.join('captoolkit', '*.py')),
)
