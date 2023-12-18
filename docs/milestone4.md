# Milestone 4

## Software Organization 

We will have our directory structured as follows: 
```
team21_2023/
│
├── docs/                 # Documentation files
│   ├── milestone4.md
│   └── ...		# other milestone documents and relevant documentation 
│
├── API_draft/                 # Describes API structure
│   ├── API_Diagram.png
│   └── README.md		
│
├── .github/
│   └── workflows/        # GitHub Actions workflows (hidden)
│       └── coverage.yml
│       └── test.yml
│
├── src/                  # Source code
│   ├── fetch_data.py
│   └── preprocess_data.py
│   └── extract_metadata.py
│   └── wavelength_alignment.py
│   └── visualize_data.py
│   └── augment_data.py
│   └── ml_model.py
│   └── interactive_visual.py
│
├── tests/                # Test suite
│   ├── test_fetch_data.py
│   └── test_preprocess_data.py
│   └── test_extract_metadata.py
│   └── test_wavelength_alignment.py
│   └── test_visualize_data.py
│   └── test_augment_data.py
│   └── test_ml_model.py
│   └── test_interactive_visual.py
│
├── setup.py              # setuptools configuration
├── requirements.txt      # Project dependencies
├── README.md             # Project README file
└── .gitignore            # Git ignore file
```

Our test suite will live in the "tests" folder, which contains files that correspond to each module in our "src" folder with "test_" appended to the front of the name. This ensures ease of testing, as pytest automatically discovers and runs tests based on these conventions. 

Of note, we decided to include a "setup.py" configuration file so that we can use setuptools to distribute our package. We chose setuptools as it is known to be simple and user-friendly for developers. setuptools has a large set of features to package, distribute, and install Python projects, and we did not anticipate needing any further features to develop our project based on the modules and classes we have planned for. We also know that setuptools is very flexible, and can integrate smoothly with PyPI. This means that it would also be easy for us to upload and distribute our package on PyPI, so others could easily install our software using "pip" later on. 

Our implementation code under the "src" folder will have the following modules, classes and functions. Our "tests" modules, classes and functions will mirror this structure, but will have "test_" appended to each.

For example, the module "fetch_data.py" in the "src" folder will have a corresponding "test_fetch_data.py" module in the "tests" folder, and the class SDSSDataFetch will have a corresponding TestSDSSDataFetch test class, and the function "construct_adql_query" will have a corresponding "test_construct_adql_query" test function.


## src implementation code outline

### Module 1. fetch_data.py

#### Class: SDSSDataFetch()

This class handles retrieving data from SDSS via their API. The SDSS API allows one to construct a query using ADQL. We plan to construct this class with the following functions:

##### __init__(self)

Creates an instance with an SDSS base URL that is set to the latest data release version for spectral queries

##### construct_adql_query(self, sdss_table, columns, constraints)

Returns an ADQL query as a string based on the desired SDSS table that is input, columns you’d like to filter by, and the exact constraints on those columns.

##### fetch_data(self, adql_query)

Accepts the ADQL query in a string format
Sends a request to the SDSS API using the sdss_url and adql_query given.
Queries are constructed by adding the adql_query onto the sdss_url.
Returns the SDSS data as a pandas dataframe.

##### fetch_spectra_SDSS(self, plate, fiberID, mjd)

Accepts the plate number and the fiberID number of the SDSS spectrosconic observation.
Returns the spectral data for the specified plate and fiberID number as a numpy array.

### Module 2. preprocess_data.py

#### Class: PreprocessDataframe()

##### Function normalization()

This function takes in a pandas series and returns a normalized pandas series

##### Function outlier_removal()

This function takes in a pandas dataframe and column names that we want to consider for outlier removal. It returns a dataframe with observations with selected features that are over +- 2.5 IQR.

##### Function interpolate()

This function takes in a pandas series and returns a interpolated pandas series

##### Function red_shift_correct()

This function takes in the observed wavelength and a red shift value and returns the corrected wavelength.

### Module 3. extract_metadata.py

#### Class: MetadataExtraction()

This class contains functions to extract a variety of metadata variables from the data object

##### __init__(self, data)

##### get_mean_value(self, column)

returns mean value of column (ie, coordinates, chemical_abundance, redshifts, etc.)

##### get_median_value(self, column)

returns median value of column

##### get_std_value(self, column)

return standard deviation of column

##### get_min_value(self, column)

returns minimum value of column

##### get_max_value(self, column): returns maximum value of column

##### extract_identifiers(self)

returns list of identifiers

##### extract_coordinates(self)

returns the right ascension ‘ra’ and declination ‘dec’ of the data

##### extract_chemical_abundances(self)

returns a dictionary of chemical abundance values

##### extract_redshifts(self)

returns a list of redshift values

##### extract_custom_metadata(self, variable_name)

will return specific metadata based on the input string if it is found in the data

### Module 4. wavelength_alignment.py

#### Class: LinearRegression()

We use the LinearRegression() class to interpolate the wavelength based on features in the data. We plan to construct the following functions/methods:

##### __init__(self)

initializes a linear regression instance.

##### fit(self, X_train, y_train)

where X_train is the training data containing the features and y_train is the wavelength outcome in the train set.

##### predict(self, X)

where X is the features used to predict the wavelength outcome.

##### get_parameters(self)

outputs the coefficients and their corresponding confidence intervals from the linear regression model.

#### Class: AstroClassification()

We use the AstroClassification() class to classify between stars, galaxies, and QSOs based on features in the data set.

##### __init__(self)

initializes a classification instance.

##### train(self, X_train, y_train)

where X_train is the training data containing the features and y_train is the astronomical label in the train set.

##### classify(self, X)

where X is the features used to classify the astronomical labels.

##### confusion_matrix(self, X)

where X is the features used to plot the confusion matrix.

##### accuracy(self, X, y)

where X is the features used to predict the labels and y is the true labels.

### Module 5. visualize_data.py

#### Class: PCAVisualization()

We use the PCAVisualization() class to implement a PCA on the data features and plot principal components against each other. We plan to construct the following functions/methods:

##### __init__(self, X)

initializes a visualization instance where X is the feature matrix.

##### plot(self)

outputs a plot of subplots containing each principal component against others.

#### Class: ClusterVisualization()

We use the ClusterVisualization() class to visualize the wavelength against each feature with clusters, distinguished using different colors for different astronomical labels, of data points. We plan to construct the following functions/methods:

##### __init__(self, X, y)

initializes a visualization instance where X is the feature matrix and y is the true or predicted labels.

##### plot(self)

outputs a plot of subplots containing wavelength against each feature with colored clusters.

### Module 6. augment_data.py

#### Class DataAugmentation()

##### __init__(self, X)

initializes DataAugmentationClass instance with data X as the preprocessed data.

##### compute_derivative(self, order=1)

computes the n_int_th derivative

##### compute_frac_derivative(self, order=0.5)

calculates  the fractional derivative of the spectrum

##### augment_data(self)

augments original X matrix with its derivatives (standard and fractional) and returns the concatenated spectral data with derivatives

##### process_spectrum(lambda)

static method, processes individual spectrum data and can be customized based on specific datasets

### Module 7. ml_model.py

#### Class MulticlassLgbmClf()

##### __init__(self, X)

initializes a LightGBM Classifier. It takes in arguments such as learning rate and number of estimators and initializes a LightGBM classifier.

##### fit(self,X,y)

takes in the training features and the target class indicating Stars, Galaxies, and QSOs.

##### predict(self,X,y)

takes a set of features and targets, performs inference, and returns a binary prediction for each of the 3 classes.

##### predict_proba(self,X,y)

takes in a set of features and targets, performs inference, and returns a probability for each class. .

##### score(self,y_true,y_pred)

will take in true target class labels and predicted target class labels and print a classification report.

### Module 8. interactive_visual.py

#### Class InveractiveViz()

##### __init__(self, X)

initializes the InteractriveViz instance, where X is the spectral data to be visualized

##### select_region(self)

allows interactive selection of regions on the spectral visualization for analysis

##### calculate_flux(self, region)

quantifies the flux within the selected region of the spectrum. This method can utilize the processed data from DataAugmentation() class

##### zoom(self)

provides zoom functionality for detailed examination of specific plot regions

##### pan(self)

provides the panning functionalize across the plot

##### overlay_spectra(self, additional_data)

allows overlaying of multiple spectral datasets for comparison

##### customize_viz(self, **kwargs)

facilitates the customization of plotting aesthetics, such as color, label, and scale

##### export_data(self)

exports the selected region, flux variables, or the entire plot as needed
show(self) renders and displays the interactive plot

## Licensing

We opted for GPL 2.0 because it upholds the principles of free and open-source software, safeguarding both user and developer rights. As a form of copyright law, GPL 2.0 ensures that software remains freely accessible, allowing for unrestricted modification and distribution. The choice of a copyleft license was driven by our commitment to keeping the software open and accessible, preventing its transformation into closed, proprietary software. Comparing GPL 2.0 and 3.0 versions, we favored 2.0 due to its extensive usage and established compatibility with a wide array of projects, facilitating seamless integration with various licenses.
