# API Update

## Fetch Data

We changed the fetch data module to allow pulling both meta data and spectral data from SDSS data base, whereas the previous fetch data API only considered the meta data of the observations. This API change tuned our features and functionalities of data acquisition to closely follow the requirements described in Annex A of the contract. This API change could also allow the subsequent modules such as machine learning module where both meta data and spectral data of the observations are needed. 

## Wavelength Alignment

We changed the wavelength alignment module to allow wavelength alignment performed on the spectral data object created through the fetch data module above. This API change made the implementation of wavelength alignment to be consistent with the fetch data module and allowed users to directly input a list of spectral data objects created by class written in fetch data module, which could be more convenient for users.

## Visualization

We changed the visualization module to focus more on wavelength and flux data and allow plotting wavelength against flux and plotting wavelength against time, which more aligned with the requirements in the contract. This API change also allowed to plot multiple observations in the same graph. This is based on the rationale that the users might want to compare different observations. 

## Preprocess

Preprocess is now a class instead of of seperate function. We implemented a class to allow storage of class attributes belong to the same observation such as the post-normalized, and post-outlier-removed flux. The outlier removal function now focuses on removing negative flux which likely represent artifacts in spectral data. It will also remove outliers based on a pre-defined z threshold. For the class method that correct redshifts,because of increased knowledge and understanding of the domain, it is now implemented to correct observed wavelength to expected wavelength based on a redshift factor z that is provided by the user. Moreover, for interpolation, a class method now implemented that takes in a reference observation and maps the flux of the orignal spectra to that of the reference spectra.

## Extract Metadata

We changed the extract metadata module to allow a user to extract equivalent widths instead of chemical abundance, as specified in the latest changes to the SRS. We added functionality to extract class information, as specified in the SRS update as well. We also eliminated functions for mean, median, min, max, and std values, as it became clear that the function of the metadata module is to retrieve columns of metadata rather than summarize a given metadata column. Our updated extract metadata module allows users to extract identifiers, coordinates, class, equivalent widths, redshift values, and any other custom metadata column the user inputs given that it exists in the data. We also updated the get_identifiers function to accept a list of identifiers to return.

## Machine Learning

The underlying machine learning framework changed from lightGBM to sklearn.RandomForestClassifier() because of lower dependencies and overheads. The API was pretty much the same. The only thing that changed was the evaluate class function now will return a confusion matrix.
