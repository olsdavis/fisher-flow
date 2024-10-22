#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="fisher-flow",
    version="1.0.0",
    description="Fisher Flow Matching official implementation",
    author="Oscar Davis, Samuel Kessler, Joey Bose",
    author_email="oscar.davis@cs.ox.ac.uk",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
