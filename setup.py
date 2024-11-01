from setuptools import setup, find_packages

setup(
    name="DynamicCellTree",
    version="0.1.0",
    description="A library for dynamic cell analysis using connected components and vector field processing",
    author="Maximilian Kotz",
    url="https://github.com/yourusername/DynamicCellTree",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "numba",
    ],
    python_requires=">=3.9",
)
