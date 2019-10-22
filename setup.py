import os
import setuptools
from glob import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='captoolkit',
    version="0.1.0",
    license='Apache',
    author='Fernando Paolo and Johan Nilsson',
    author_email='paolofer@jpl.nasa.gov, johan.nilsson@jpl.nasa.gov',
    description='JPL Cryosphere Altimetry Processing Toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/fspaolo/captoolkit',
    packages=['captoolkit', 'captoolkit.lib'],  #NOTE: Check this when inclusion of lib
    #packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    #NOTE 1: Install all python scripts as stand-alone command line utils.
    #NOTE 2: If Anaconda Python, executable scripts will be placed
    # at ~/anaconda3/bin, otherwise the default is /usr/local/bin 
    scripts=glob('captoolkit/*.py'), 
)
