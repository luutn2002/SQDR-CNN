import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="sqdr-cnn",
    py_modules=["sqdr_cnn"],
    version="0.0.1",
    description="Source code of SQDR-CNN.",
    author="LUU Trong Nhan",
    author_email = "ltnhan0902@gmail.com",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    extras_require={'dev': ['torchquantum']}
)