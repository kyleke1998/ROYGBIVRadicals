import scipy.interpolate as interp
import numpy as np
from fetch_data_spectra import SDSSDataFetchSpectra as SDSSDataFetch

class align_wavelength():
    """
    Aligns the wavelength for the specified object from the Sloan Digital Sky Survey (SDSS) data base.

    Attributes
    ----------
    obs : list of SDSSDataFetch objects
        The spectral data objects of the class SDSSDataFetch() to be aligned.
    ref : SDSSDataFetch
        The reference spectral data object of the class SDSSDataFetch().
    ref_flux : dict
        The flux data for reference spectral data object.
    ref_wl : dict
        The wavelength data for reference spectral data object.
    interp : dict
        The interpolate function for each wavelength type. 
    wl_name : str
        The name of wavelength type to be aligned.

    Methods
    -------
    __init__(self, obs_spectra, ref_spectra):
        Initializes the class instance.
    interpolate(self):
        Creates the interpolation function for each wavelength.
    align(self):
        Returns the aligned wavelength data of the spectral data objects.
    """

    def __init__(self, obs_spectra, ref_spectra):
        """
            Initializes the class instance. 

            Parameters
            ----------
            obs_spectra : list
                The list of spectral data objects of class SDSSDataFetch() to be aligned. 
            ref_spectra : SDSSDataFetch
                The reference spectral data object of class SDSSDataFetch().

            Returns
            -------
            None
        """
        # store observed spectral data object
        # obs spectral data could be a list of spectral objects
        self.obs = obs_spectra
        self.ref = ref_spectra
        self.obs_align = []
        self.ref_flux = ref_spectra.get_flux()
        self.ref_wl = ref_spectra.get_wavelength()
        # set the interpolation object
        self.interp = {}
        # get the wavelength names from spectral object
        self.wl_name = None
  
    def interpolate(self, wl_name):
        """
            Constructs the interpolation function. 
            
            Parameters
            ----------
            wl_name : str
                The name of the wavelength type to be aligned.

            Returns
            -------
            None
        """
        # check whether obs and reference have the same wavelength names
        if wl_name not in list(self.ref_flux.keys()) or wl_name not in list(self.ref_wl.keys()):
            raise ValueError('The wavelength types of the reference spectral object and the spectral object to be aligned do not match.')
        self.wl_name = wl_name
        self.interp[wl_name] = interp.interp1d(self.ref_flux[wl_name], self.ref_wl[wl_name], bounds_error = False, kind = 'linear', fill_value = 'extrapolate')
  
    def align(self):
        """
            Aligns the spectral data specified.

            Returns
            -------
            obs : list
                The list of spectral data objects of class SDSSDataFetch with aligned wavelength stored.
        """
        wl = self.wl_name
        if len(self.interp) == 0:
            raise NotImplementedError('Please first create interpolation function using the method interpolate().')

        for idx, obs in enumerate(self.obs):

            obs_wl = obs.get_wavelength()
            obs_flux = obs.get_flux()
            obs_wl_names = list(obs_wl.keys())
            plate = obs.plate
            fiberID = obs.fiberID
            mjd = obs.mjd
            dr = obs.dr
            
            obs_new = SDSSDataFetch(dr, plate, fiberID, mjd)

            obs_flux_reshape = np.linspace(min(obs_flux[wl]), max(obs_flux[wl]), self.ref_flux[wl].size)
            obs_wl_align = self.interp[wl](obs_flux_reshape)
            obs_new.set_wavelength(wl, obs_wl_align)
            obs_new.set_flux(wl, obs_flux_reshape)
        
            self.obs_align += [obs_new]

        return self.obs_align