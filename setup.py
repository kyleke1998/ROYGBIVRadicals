from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name="roygbivradicals_2023",
    version="2.3",
    packages=find_packages(where='src'),
    package_dir={'':'src'},
    install_requires=required_packages,
    author="Group 21 in CS207 2023 Cohort (Clare, Carrie, Kyle, Kevin, Abbie)",
    description="Software assisting with with astronomical research interfacing directly with the Sloan Digital Sky Survey (SDSS) services.",
    url="https://code.harvard.edu/CS107/team21_23",
    license = "GPL-2.0", 
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
