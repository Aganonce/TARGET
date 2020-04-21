#!/usr/bin/env python

from distutils.core import setup

setup(name='target',
    version='1.0.0',
    description='Text-Agnostic Response-Generated Event Tracking',
    author='James Flamino',
    author_email='flamij@rpi.edu',
    packages=['tools'],
    install_requires = ['numpy', 'sklearn', 'networkx>=1.9.1,<2.4', 'pandas', 'hdbscan', 'matplotlib', 'dask[complete]']
)
