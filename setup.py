#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="simplex-flow",
    version="0.0.5",
    description="Simplex Flow Matching official implementation",
    author="",
    author_email="",
    url="https://github.com/olsdavis/simplex-flow",
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
