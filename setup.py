#!/usr/bin/env python3
"""
Setup script for hex_quality package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Hexahedral Element Quality Metrics package"

setup(
    name="hex_quality",
    version="1.0.0",
    author="Finite Element Quality Assessment",
    description="Quality metrics for 8-node hexahedral finite elements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "examples": [
            "matplotlib>=3.0",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="finite elements, mesh quality, hexahedral, FEM, quality metrics",
    project_urls={
        "Source": "https://github.com/ZhuoQiuMcgill/hexquality",
        "Bug Reports": "https://github.com/ZhuoQiuMcgill/hexquality/issues",
        "Documentation": "https://github.com/ZhuoQiuMcgill/hexquality/blob/main/README.md",
    },
)