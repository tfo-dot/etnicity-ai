#!/usr/bin/env python

from setuptools import setup, find_packages


name = 'etnicity_ai'
version = '1'
description = 'Etnicity AI'
author = ''

setup(
      name=name,
      version=version,
      description=description,
      author=author,
      packages=find_packages(where='src'),
      package_dir={'':'src'},
)