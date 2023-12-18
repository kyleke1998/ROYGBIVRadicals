import pandas as pd
import numpy as np


class SDSSDataFetchSpectra():
    """
    Fetches the spectrum and meta data for individual specified observation from the Sloan Digital Sky Survey (SDSS) data base.

    Attributes
    ----------
    dr : int
        The data release version of SDSS
    plate : int
        The plate number of the SDSS spectroscopic observation.
    fiberID : int
        The fiberID number of the SDSS spectroscopic observation.
    mjd : int
        The Modified Julian Date (MJD) of the SDSS spectroscopic observation.
    meta : pandas.DataFrame
        The meta data of the SDSS spectroscopic observation.
    wavelength : dict
        The wavelength of the SDSS spectroscopic observation.
    flux : dict
        The flux of the SDSS spectroscopic observation.

    Methods
    -------
    __init__(self, dr_release, plate, fiberID, mjd):
        Initializes the class instance.
    get_wavelength(self):
        Returns the wavelength of the SDSS spectroscopic observation.
    set_wavelength(self, wl_name, new_wl):
        Sets the wavelength of the SDSS spectroscopic observation.
    get_flux(self):
        Returns the flux of the SDSS spectroscopic observation.
    set_flux(self):
        Sets the flux of the SDSS spectroscopic observation.
    get_meta(self):
        Returns the meta data of the SDSS spectroscopic observation.
    fetch(self):
        Fetches data of the SDSS spectroscopic observation from SDSS.
    """

    def __init__(self, dr_release, plate, fiberID, mjd):
        """
            Initializes the class instance. 

            Parameters
            ----------
            dr_release : int
                The data release version of SDSS. Must be DR8 or later.
            plate : int
                The plate number of the SDSS spectroscopic observation.
            fiberID : int
                The fiberID number of the SDSS spectroscopic observation.
            mjd : int
                The Modified Julian Date (MJD) of the SDSS spectroscopic observation.

            Returns
            -------
            None
        """
        self.plate = plate
        self.fiberID = fiberID
        self.mjd = mjd
        self.dr = dr_release
        # initialize meta data, flux, and wavelength
        self.meta = None
        self.wavelength = {}
        self.flux = {}

        if dr_release < 8:
            raise ValueError('The data release version must be DR8 or later.')

    def get_wavelength(self):
        """
            Returns the retrieved wavelength of the SDSS spectroscopic observation. 

            Returns
            -------
            wavelength : dict
                The dictionary of wavelength data of specified spectral object containing all the wavelength types.
        """
        if len(self.wavelength) == 0:
            raise NotImplementedError('Please first fetch or set the corresponding data from SDSS by using the method fetch().')
        return self.wavelength

    def set_wavelength(self, wl_name, new_wl):
        """
            Sets the wavelength data for the SDSS spectroscopic observation. 

            Parameters
            ----------
            wl_name : str
                The wavelength name to be updated. 
            new_wl : numpy.array
                The new wavelength value.

            Returns
            -------
            None
        """
        if wl_name in self.wavelength:
            if len(self.wavelength[wl_name]) != len(new_wl):
                raise ValueError('Please enter a new wavelength data the same length as the old data.')
        if isinstance(new_wl, np.ndarray) == False:
            raise TypeError('Please enter a numpy array for new wavelength data.')

        self.wavelength[wl_name] = new_wl
  
    def get_flux(self):
        """
            Returns the retrieved flux of the SDSS spectroscopic observation. 

            Returns
            -------
            flux : dict
                The dictionary of flux data of specified spectral object containing all the wavelength types.
        """
        if len(self.flux) == 0:
            raise NotImplementedError('Please first fetch or set the corresponding data from SDSS by using the method fetch().')
        return self.flux

    def set_flux(self, wl_name, new_flux):
        """
            Sets the flux data for the SDSS spectroscopic observation. 

            Parameters
            ----------
            wl_name : str
                The wavelength name to be updated.
            new_flux : numpy.array
                The new wavelength value.

            Returns
            -------
            None
        """
        if wl_name in self.flux:
            if len(self.flux[wl_name]) != len(new_flux):
                raise ValueError('Please enter a new flux data the same length as the old data.')
        if isinstance(new_flux, np.ndarray) == False:
            raise TypeError('Please enter a numpy array for new flux data.')

        self.flux[wl_name] = new_flux
  
    def get_meta_data(self):
        """
            Returns the retrieved meta data of the SDSS spectroscopic observation. 
            Meta data includes ra, dec, objid, run, reun, camcol, field, z, plate, mjd, fiberID, specobjid, run2d, and equiv_widths.

            Returns
            -------
            meta : pandas.DataFrame
                The meta data of specified spectral object.
        """
        if self.meta is None:
            raise NotImplementedError('Please first fetch or set the corresponding data from SDSS by using the method fetch().')
        return self.meta 
  

    def fetch(self):
        """
            Fetches the data (both meta data and spectrum) from Sloan Digital Sky Survey (SDSS).

            Returns
            -------
            None
        """
        from astroquery.sdss import SDSS
        pull_meta = SDSS.query_specobj(plate = self.plate, fiberID = self.fiberID, mjd = self.mjd, data_release = self.dr)
        pull_spec = SDSS.get_spectra(plate = self.plate, fiberID = self.fiberID, mjd = self.mjd, data_release = self.dr)[0]
        self.meta = pull_meta.to_pandas()
        self.meta = self.meta.assign(equiv_widths=pd.Series([{}] * len(self.meta)))
        for idx, spec in enumerate(pull_spec):

            if idx == 3: 
                # Access and append equivalent widths to meta dataframe
                equiv_width_data = spec.data['lineew']
                line_name_data = spec.data['linename']
                # Have line names as keys and equivalent width data as values
                self.meta.at[0, 'equiv_widths'] = dict(zip(line_name_data, equiv_width_data))
	    
            if idx in [1, 4, 5, 6, 7, 8, 9]:
                name = spec.name

                flux_data = spec.data['flux']
                wl_data = spec.data['loglam']

                self.flux[name] = flux_data
                self.wavelength[name] = wl_data