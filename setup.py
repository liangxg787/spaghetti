# -*- coding: UTF-8 -*-
"""
@Time : 15/06/2025 10:52
@Author : Xiaoguang Liang
@File : setup.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
from setuptools import setup, find_packages
from spaghetti import VERSION

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="spaghetti",
    version=VERSION,
    author='Xiaoguang Liang',
    author_email='hplxg@hotmail.com',
    url='https://github.com/liangxg787/spaghetti.git',
    description='SPAGHETTI: Editing Implicit Shapes Through Part Aware Generation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    keywords='spaghetti',
    include_package_data=True,
    zip_safe=False,
    packages=["spaghetti"],
    install_requires=[],
    entry_points={},
    classifiers=[
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
    ]
)
