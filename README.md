
[![.github/workflows/test.yml](https://code.harvard.edu/CS107/team21_2023/actions/workflows/test.yml/badge.svg?branch=main)](https://code.harvard.edu/CS107/team21_2023/actions/workflows/test.yml?query=branch%3Amain)

[![.github/workflows/coverage.yml](https://code.harvard.edu/CS107/team21_2023/actions/workflows/coverage.yml/badge.svg?branch=main)](https://code.harvard.edu/CS107/team21_2023/actions/workflows/coverage.yml?query=branch%3Amain)

# CS107 Final Project - Group #21

This repository contains group #21's final project for CS107/AC207.

## Group Members

Clare Morris, Kyle Ke, Kevin Liu, Carrie Cheng, Abbie Kinaro

## Installation
Follow steps below for proper installation from [TestPyPI](https://test.pypi.org/). 
1. Create an isolated virtual environment. This is optional but recommended for this installation: `python3 -m venv myenv`
2. Activate the virtual environment : `source myenv/bin/activate`
3. Install the package using pip from TestPyPI: `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple roygbivradicals_2023==2.3`


*Remember to replace 'myenv' with your chosen environment name.

## Documentation

You can find the documentation on this package in the [docs](./docs) folder.

## Overview
We have created 7 modules in the package. A rough descripion of each can be found below. You can find more infomation on each module in the [API_draft](./API_draft) folder.

### Module 1: fetch_data.py
This module manages the retrieval of spectral data from the Sloan Digital Sky Survey (SDSS) via their API. It provides classes like `SDSSDataFetchSQL` and `SDSSDataFetchSpectra` that facilitate querying SDSS for specific data using ADQL queries. These classes handle the construction of queries, fetching data based on specified constraints, and organizing the retrieved data into pandas dataframes, allowing easy manipulation and analysis.

### Module 2: preprocess.py
This module contains the `Preprocess` class, which processes observed spectral data. It offers methods for normalization, outlier removal, interpolation, and redshift correction. These operations enable the transformation and optimization of raw spectral data for downstream analysis.

### Module 3: extract_metadata.py
The `MetadataExtraction` class in this module specializes in extracting various metadata attributes from spectral data objects. It provides methods to retrieve identifiers, coordinates, classification, redshift values, equivalent widths, and custom metadata, organizing these details for comprehensive analysis and categorization.

### Module 4: align_wavelength.py
This module facilitates aligning wavelength data among spectral observations. Through the `align_wavelength` class, it constructs interpolation functions to align observed spectral data with a reference, ensuring consistency in wavelength across multiple observations.

### Module 5: visualization.py
Designed for visualizing spectral data, this module houses the `spectra_visualization` class. It converts spectral data into pandas dataframes and offers methods to plot and visualize the wavelength-flux relationships, enabling users to observe and interpret spectral characteristics effectively.

### Module 6: augment_data.py
The `DataAugmentation` class within this module focuses on augmenting spectral data. It performs operations like computing derivatives and fractional derivatives, enhancing the dataset for more nuanced analysis and feature extraction.

### Module 7: machine_learning.py
This module contains the `RandomForestClassifierWrapper` class, which wraps a Random Forest Classifier. It provides methods for fitting the classifier, applying Principal Component Analysis (PCA), performing predictions, and evaluating model performance, enabling classification tasks on spectral data. This module focuses on machine learning.

### Module 8: interactive_visual.py
This module focuses on interactive visualizations for spectral data analysis. The `InteractiveViz` class offers functionalities for interactive selection of spectral regions, flux quantification, zooming, panning, overlaying spectra, and customizing plot aesthetics, enhancing the user's exploration and understanding of spectral data patterns.

## Impact Statement
We have developed this software to create a streamlined integration with Sloan Digital Sky Survey (SDSS) services, providing direct access to spectral data and related information in hopes of advancing astronomical research and actions. Its core advantage lies in its comprehensive suite of functionalities designed to streamline data analysis within the realm of celestial objects. By seamlessly integrating with the SDSS services, the software provides direct access to spectral data and associated information, freeing researchers from the intricacies of data retrieval and allowing them to delve deeply into analysis.

At its heart, the software's strength lies in its advanced capabilities, spanning from intricate data preprocessing to metadata extraction and spectral alignment. This arsenal empowers researchers with precision and efficiency, enabling them to extract profound insights from astronomical data. Moreover, its modular architecture fosters adaptability, allowing users to tailor analyses to their specific research needs. This customization potential, combined with an extensible framework that accommodates additional features, positions the software as a versatile ally for a wide spectrum of astronomical investigations.

What sets this software apart is the fact that it is easy to understand from documented practices, and allows for collaoration. We believe that this software encourages interdisciplinary research, and promises to be a resilient and evolving asset in the ever-changing landscape of astronomical exploration.

### Contributions
#### Name: Kyle Ke, NetID: sik456
*Contributions*: I spent approximately 40 hours on the project. I worked on the preprocess and machine learning modules - both implementation and unit test. I also assisted in the CI/CD yaml files and the integration tests.

#### Name: Carrie Cheng, NetID: zhc363
*Contributions*: I dedicated 65 hours on the project. I worked on the implementation and unit testing of the fetch_data_meta, fetch_data_spectra, wavelength alignment, and visualization modules. I also worked on the corresponding integration tests on these four modules. I worked on the video presentation, fixed bugs, finalized the package code, assisted in uploading the package to test pypi, and updating the tutorial notebook with the installation from test pypi and tested it worked for each module. 

#### Name: Abbie Kinaro, NetID: abk110
*Contributions*: I've spent roughly 25 hours on the project. I've worked on the readme files, ymlfiles, license and setuptools script.

#### Name: Clare Morris, NetID: clm693
*Contributions*: I dedicated approximately 50 hours to this project. I developed the source code, unit tests, and integration tests for the extract metadata module as well as code for finding equivalent width data via API. I also wrote up documentation for software organization, created our API diagram, set up our initial code repo, and worked on setting up our library on TestPyPI. I also worked on setting up the ymlfiles to run tests and code coverage on pushes to/from dev, and spent time troubleshooting the badges for these files in the README. Lastly I served as project lead and set up meeting times, led and assigned tasks in meetings, and created meeting agendas.

#### Name: Kevin Liu, NetID: kel166
*Contributions*: I spent approximately 30 hours on this project. I developed the source code, unit tests, and integration tests for the DataAugmentation, and InteractiveViz modules.
