from setuptools import setup, find_packages

setup(
    name="effconnpy",  # Replace with your package name
    version="0.1.24",  # Initial version
    description="A package for causal inference and statistical modeling in brain time series",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # Use Markdown for the README
    author="Alessandro Crimi",
    author_email="alecrimi@agh.edu.pl",
    url="https://github.com/alecrimi/effconnpy",  # Link to your project
    packages=find_packages(),  # Automatically find packages
    install_requires=[  # Dependencies
        "numpy",
        "pandas",
        "statsmodels",
        "scipy",
        "copent",
        "networkx",
        "dowhy",
        "pymc3",
        "arviz",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",  # Minimum Python version
    license="MIT",  # License type
)

