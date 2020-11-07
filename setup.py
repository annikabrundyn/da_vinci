#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='da_vinci',
    version='0.0.0',
    description='DaVinci Depth Estimation + Stereo View Prediction',
    author='',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/annikabrundyn/da_vinci',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)