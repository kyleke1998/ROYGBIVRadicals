# Pipeline Outline

### Module 1. fetch_data.py

#### Class: SDSSDataFetchSQL()

This class handles retrieving meta data from SDSS via their API. The SDSS API allows one to construct a query using ADQL. We plan to construct this class with the following functions:

##### __init__(self)

Creates an instance with an SDSS base URL that is set to the latest data release version for spectral queries.

##### construct_adql_query(self, sdss_table, columns, constraints)

Returns an ADQL query as a string based on the desired SDSS table that is input, columns you’d like to filter by, and the exact constraints on those columns.

##### fetch_data(self, adql_query, dr)

Accepts the ADQL query in a string format and data release version in interger format. Sends a request to the SDSS API using the sdss_url and adql_query given. Queries are constructed by adding the adql_query onto the sdss_url. Returns the SDSS data as a pandas dataframe.

#### Class: SDSSDataFetchSpectra()

This class handles retrieving spectral data from SDSS via their API. We plan to construct this class with the following functions:

##### __init__(self, dr_release, plate, fiberID, mjd)

Creates an instance with an SDSS base URL that is set to the user defined data release version through the parameter dr_release and user defined SDSS observation through the parameters plate, fiberID, and mjd (Modified Julian Date) for spectral queries. In this initialization, we store the data release version, plate, fiberID, mjd input by the users in the object attributes (self.dr, self.plate, self.fiberID, self.mjd). We initialize empty pandas data frame for instance meta data (self.meta), empty dictionary for instance wavelength data (self.wavelength), and empty dictionary for instance flux data (self.flux).

##### get_wavelength(self)

Returns the wavelength data of the SDSS spectroscopic observation in a dictionary containing the wavelength data for each wavelength type.

##### set_wavelength(self, wl_name, new_wl)

Accepts the name of the wavelength type (wl_name) in the string format and the wavelength data (new_wl) in the numpy array format. Stores the input wavelength data into the object. If there is existing wavelength data stored in the object, the input wavelength data will replace the existing data.

##### get_flux(self)

Returns the flux data of the SDSS spectroscopic observation in a dictionary containing the flux data for each wavelength type.

##### set_flux(self, wl_name, new_flux)

Accepts the name of the wavelength type (wl_name) in the string format and the flux data (new_flux) in the numpy array format. Stores the input flux data into the object. If there is existing flux data stored in the object, the input flux data will replace the existing data.

##### get_meta_data(self):

Returns the meta data of the SDSS spectroscopic observation in a pandas data frame.

##### fetch(self):

Fetches and queries the corresponding SDSS spectroscopic observation data including the spectral data (wavelength and flux) and the meta data based on the user specified spectral observation identifier at the instance initialization from SDSS data base. Stores the wavelength and flux data in a dictionary for each wavelength type within the object. Could use get_flux() and get_wavelength() method to output the flux and wavelength data for this instance respectively.


### Module 2. preprocess.py

#### Class: Preprocess()

##### __init__(self, wavelength, flux)

Initializes an instance of Preprocess class. It initializes class attributes: observed wavelength and flux data (self.flux_wl_data), normalized flux data (self.normalized_flux_wl_data), outlier removed flux data (self.outlier_removed_flux_wl_data), redshift corrected flux data (self.redshift_corrected_flux_wl_data), and redshifted flux data as class attributes.

##### normalization(self, method="minmax")

Normalizes the raw flux data for a given input method. Currently, only the "minmax" method is accepted. Returns the wavelength and normalized flux in a 2-D numpy matrix.

##### remove_outliers(self, z_threshold=10)

Removes negative values in flux and removes flux that is greater than a certain Z-score threshold (z_threshold). Returns the outlier-removed flux and corresponding wavelength in a 2-D numpy array.

##### interpolate(self,ref_flux,ref_wl,kind="linear")

This function takes in as arguments the reference wavelength (ref_wl) and reference flux (ref_flux). It initialize the scipy interp1d function with observed wavelength and observed flux, it then interpolates the observed flux to have the same length as the reference wavelength. The function returns a 2-D numpy array with interpolated flux and reference wavelength.

##### correct_redshift(self,redshift)

This function takes a red shift value and returns a 2-D numpy array with the expected flux and corresponding wavelength.

### Module 3. extract_metadata.py

#### Class: MetadataExtraction()

This class contains functions to extract a variety of metadata variables from the data object

##### __init__(self, data)

##### get_identifiers(self, identifier_columns)

returns dataframe with identifiers as column names

##### get_coordinates(self)

returns dataframe with right ascension ‘ra’ and declination ‘dec’ as column names

##### get_class(self)

returns a series with classes for each data object (each row)

##### get_redshifts(self)

returns redshift values, 'z', as a series

##### get_equiv_widths(self)

returns a series that contains the equivalent width information for each data object (each row)

##### get_custom_metadata(self, variable_name)

will return specific metadata columns based on the input string if they are found in the data

### Module 4. align_wavelength.py

#### Class: align_wavelength()

We use the align_wavelength() class to interpolate the wavelength based on the reference spectral data. We plan to construct the following functions/methods:

##### init(self, obs_spectra, ref_spectra)

Initializes a wavelength alignment instance. Accepts a list of spectral data objects of class SDSSDataFetch() for spectral data objects to be aligned with (obs_spectra) the reference spectral data object, and the spectral data object of class SDSSDataFetch() that serves as the reference (ref_spectra). In this initialization, we store the obs_spectra (self.obs) and ref_spectra (self.ref) input by the users and extract the wavelength data (self.ref_wl), flux data (self.ref_flux), and initialize the specified wavelength type name (self.wl_name) to be aligned with None. We initialize empty dictionary for interpolation functions (self.interp)

##### interpolate(self, wl_name)

Creates and constructs the interpolation functions based on the reference spectral data object. An interpolation function is created for the specified wavelength type through the parameter wl_name. Stores the constructed interpolation functions in the object attribute (self.interp).

##### align(self)

Aligns the spectral data specified by the users with the reference spectral data object. Utilizes the constructed interpolation functions to align the wavelength data of each observation specified in the list specified by users with the reference spectral data. Returns a list of spectral data objects of class SDSSDataFetch() containing the aligned wavelength for corresponding spectral observation. This list retains the same order of spectral observation objects as in the list of spectral data objects (in parameter obs_spectra) specified by the users when initializing the instance.

### Module 5. visualization.py

#### Class: spectra_visualization()

We use the spectra_visualization class to implement visualizations on the SDSS spectrosconic data. We plan to construct the following functions/methods:

##### init(self, spectra_data, wl_name)

Initializes a visualization instance. Accepts a list of spectral data objects of class SDSSDataFetch (spectra_data) to plot visualization on and the wavelength type to be plotted (wl_name). In this initialization, we store spectra_data (self.data) and wl_name (self.wl_plot) input by the users in the object attributes. We initialize empty data frame (self.df) for the input spectral data objects with None, empty size of wavelength data (self.len_wl) for the input spectral data objects (all objects should have the same size of wavelength data) with None, and unique spectral identifiers combining plate, fiberID, and mjd (self.unique_spec_ids) of the input spectral data objects.

##### convert_pd(self)

Converts and the list of specified spectral data objects of class SDSSDataFetch to a pandas data frame containing the spectral data for all observations. Stores the converted pandas data frame in object attribute (self.df). Returns a pandas data frame containing the data for all SDSS spectroscopic observations specified by the users containing the spectral data (wavelength, flux, plate, fiberID, Modified Julian Date (mjd), data releaser version, time, and a spectral object identifier (combining plate, fiberID, and mjd).

##### viz_flux_wavelength(self)

Returns a scatter plot of wavelength against flux for all spectral objects specified by users.

##### viz_wavelength(self, plate, fiberID, mjd)

Accepts information of the observation through the plate, fiberID, and Modified Julian Date (mjd) of the spectral object to be visualized on. Returns a plot of wavelength against time for the specified spectral object.

### Module 6. augment_data.py

#### Class DataAugmentation()

##### __init__(self, X)

Initializes DataAugmentationClass instance with data X as the preprocessed data.

##### compute_derivative(self, order=1)

computes the n_int_th derivative

##### compute_frac_derivative(self, order=0.5)

calculates  the fractional derivative of the spectrum

##### augment_data(self)

augments original X matrix with its derivatives (standard and fractional) and returns the concatenated spectral data with derivatives

##### process_spectrum(lambda)

static method, processes individual spectrum data and can be customized based on specific datasets

### Module 7. machine_learning.py

#### Class RandomForestClassifierWrapper()

##### __init__(self, X)

Initializes a Random Forest Classifier instance with the specified hyperparameters. The constructor initializes class attributes such as the number of estimators (self.n_estimators), criterion for splitting nodes (self.criterion), maximum depth of the tree (self.max_depth), minimum samples required to split an internal node (self.min_samples_split), minimum samples required to be a leaf node (self.min_samples_leaf), minimum weighted fraction of the sum total of weights (self.min_weight_fraction_leaf), maximum number of features to consider for a split (self.max_features), maximum number of leaf nodes (self.max_leaf_nodes), minimum impurity decrease required for a split (self.min_impurity_decrease), whether to use bootstrap samples (self.bootstrap), whether to use out-of-bag samples for scoring (self.oob_score), number of jobs for parallel processing (self.n_jobs), random state (self.random_state), verbosity level (self.verbose), whether to enable warm starting (self.warm_start), class weights (self.class_weight), complexity parameter for Minimal Cost-Complexity Pruning (self.ccp_alpha), and maximum samples for bootstrapping (self.max_samples). Adjust these hyperparameters to customize the behavior of the Random Forest Classifier.
 

##### fit(self, X, y, apply_pca=True, pca_threshold=0.95)
Takes in as arguments a feature matrix (X), target class array (y) with 3 classes, whether to apply PCA (apply_pca), and a PCA explained varaince threshold (pca_threshold). It first performs dimensioanlity reduction on the feature matrix and keeps the principal components that is smaller than or equal to the explained variance threshold.

##### apply_pca(self, X, threshold=0.95)
Takes in as arguents a feature matrix (X), and PCA explained variance percentage threshold (0.95). It performs PCA and returns the number of principal components with cumalative explained variance percentage smaller or equal than 0.95 as a 2-D numpy array.

##### apply_pca_test(self, X_test)
Takes in as argument feature_matrix (X_test). Applies PCA on the test set with the fitted PCA from apply_pca. Returns the principal components for the test features as a 2-D array.


##### predict(self,X_test)

Takes in a feature matrix (X) and performs inference using the fitted model and returns a binary numpy array of the predictions.

##### predict_proba(self,X_test)

Takes in a feature matrix (X) and performs inference using the fitted model and returns a matrix with probabilities of each of the three classes.

##### evaluate(self,y_true,y_pred)

Takes in true target class labels (y_true) and predicted target class labels (y_pred) and returns a confusion matrix as list of lists.

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

