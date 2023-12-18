import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from scipy.stats import zscore

class Preprocess():
 
    """
    Preprocess spectra data of a single observation from the Sloan Digital Sky Survey (SDSS).
    This class provides methods for normalizing, removing outliers, and correcting redshift in the flux data.

    Attributes
    ----------
    flux_wl_data : np.ndarray
        2D array containing wavelength and flux data.

    normalized_flux_wl_data : np.ndarray
        2D array containing normalized flux data and corresponding wavelength.

    outlier_removed_flux_wl_data : np.ndarray
        2D array containing flux data with outliers removed and corresponding wavelength.

    redshift_corrected_flux_wl_data : np.ndarray
        2D array containing flux data with redshift corrected and corresponding wavelength.

    Methods
    -------
    __init__(self, wavelength, flux)
        Initializes the Preprocess class with wavelength and flux data.

    normalize(self, method='minmax') -> np.ndarray
        Normalize the flux data.

    remove_outliers(self, z_threshold=10) -> np.ndarray
        Remove outliers from the flux data considering non-negativity and extreme values.

    correct_redshift(self, redshift) -> np.ndarray
        Corrects redshift in the wavelength data.
    """

    def __init__(self, wavelength, flux):
        """
        Initializes the Preprocess class with wavelength and flux data.

        :param wavelength: Array of wavelength values.
        :type wavelength: np.ndarray

        :param flux: Array of flux values.
        :type flux: np.ndarray
        """
        self.flux_wl_data = np.vstack([wavelength, flux])
        self.normalized_flux_wl_data = None
        self.outlier_removed_flux_wl_data = None
        self.redshift_corrected_flux_wl_data = None

    def normalize(self, method='minmax') -> np.ndarray:
        """
        Normalize the flux data.

        :param method: Normalization method ('minmax').
        :type method: str

        :return: 2D array with normalized flux data and corresponding wavelength.
        :rtype: np.ndarray
        """
        flux_data = self.flux_wl_data[1]

        if method == 'minmax':
            scaler = MinMaxScaler()
            normalized_flux = scaler.fit_transform(flux_data.reshape(-1, 1)).flatten()
            if (normalized_flux > 1).any() or (normalized_flux < 0).any():
                raise ValueError("Normalization failed. Please check the input flux data.")

        else:
            raise ValueError("Invalid normalization method. Supported methods: 'minmax'.")

        self.normalized_flux_wl_data = np.vstack([self.flux_wl_data[0], normalized_flux])
        return self.normalized_flux_wl_data

    def remove_outliers(self, z_threshold=10) -> np.ndarray:
        """
        Remove outliers from the flux data considering non-negativity and extreme values.

        :param z_threshold: Z-score threshold for outlier removal.
        :type z_threshold: float

        :return: 2D array with outlier-removed flux data and corresponding wavelength.
        :rtype: np.ndarray
        """

        flux_data = self.flux_wl_data[1]
        wl_data = self.flux_wl_data[0]

        z_scores = zscore(flux_data)

        outlier_indices = np.abs(z_scores) > z_threshold
        outlier_indices |= flux_data < 0
        outlier_removed_flux = flux_data[~outlier_indices]
        corresponding_wavelength = wl_data[~outlier_indices]
        assert len(outlier_removed_flux) == len(corresponding_wavelength)
        self.outlier_removed_flux_wl_data = np.vstack([corresponding_wavelength, outlier_removed_flux])
        return self.outlier_removed_flux_wl_data

    def correct_redshift(self, redshift) -> np.ndarray:
        """
        Corrects redshift in the wavelength data.

        :param redshift: Redshift value for correction.
        :type redshift: float

        :return: 2D array with redshift-corrected flux data and corresponding wavelength.
        :rtype: np.ndarray
        """
        corrected_wavelength = self.flux_wl_data[0] / (1 + redshift)
        self.redshift_corrected_flux_wl_data = np.vstack([corrected_wavelength, self.flux_wl_data[1]])
        return self.redshift_corrected_flux_wl_data
    
    def interpolate(self, ref_flux, ref_wl, kind='linear'):
        """
        Interpolate flux data to a new wavelength grid using scipy interp1d.

        Parameters:
        - ref_flux (numpy.ndarray): Reference flux data to be interpolated.
        - ref_wl (numpy.ndarray): Reference wavelength grid corresponding to ref_flux.
        - kind (str, optional): Interpolation method. Default is 'linear'.

        Returns:
        -   numpy.ndarray: Reference flux stacked with interpolated flux.
        """
        interp_function = interp1d(self.flux_wl_data[0], self.flux_wl_data[1], kind='linear', fill_value='extrapolate')

        interpolated_flux = interp_function(ref_wl)

        return interpolated_flux 




