#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

setup(
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    zip_safe=False,
)
