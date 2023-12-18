import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from fetch_data_spectra import SDSSDataFetchSpectra as SDSSDataFetch
from align_wavelength import align_wavelength

class spectra_visualization():
    """
    Visualizes wavelength for the specified object from the Sloan Digital Sky Survey (SDSS) data base.

    Attributes
    ----------
    data : list
        The list of spectral objects of class SDSSDataFetch to be plotted.
    df : pandas.DataFrame
        The converted pandas data frame for the list of spectral objects.
    len_wl : int
        The size of wavelength data in each spectral object. The wavelength data of the spectral objects should be aligned.
    unique_spec_ids : list
        The list of unique spectral object identifiers with the combination of plate, fiberID, and Modified Julian Date (MJD).

    Methods
    -------
    __init__(self, spectra_data, wl_name):
        Initializes the class instance.
    convert_pd(self):
        Converts the SDSSDataFetch spectral objects to pandas data frame.
    viz_flux_wavelength(self):
        Plots the wavelength against flux.
    viz_wavelength(self, plate, fiberID, mjd):
        Plots the wavelength.
    """

    def __init__(self, spectra_data, wl_name):

        """
            Initializes the class instance. 

            Parameters
            ----------
            spectra_data : list
                The list of spectral data objects of class SDSSDataFetch to plot visualization on.
            wl_name : str
                The wavelength type to be plotted.

            Returns
            -------
            None
        """

        # store object meta data
        self.data = spectra_data
        self.df = None
        self.len_wl = None
        self.wl_plot = wl_name
        self.unique_spec_ids = []
  
    def convert_pd(self):
        """
            Converts the list of spectral data objects of class SDSSDataFetch to pandas dataframe. 

            Returns
            -------
            df_wl : pandas.DataFrame
                The converted pandas data frame from the list of spectral data objects.
        """

        df_wl = pd.DataFrame()
        wl_name = self.wl_plot

        for idx, d in enumerate(self.data):
                
            wl_data = d.get_wavelength()
            
            if wl_name not in wl_data:
                raise ValueError('The wavelength type is not included in the spectral data.')
                
            len_wl = len(wl_data[wl_name])
            flux_data = d.get_flux()
            plate = d.plate
            fiberID = d.fiberID
            mjd = d.mjd
            dr = d.dr

            if idx != 0:
                if prev_len != len_wl:
                    raise ValueError('The wavelength dimension must match for all the spectral data.')

            data_single = {wl_name: wl_data[wl_name], 
                         'flux': flux_data[wl_name],
                          'plate': [plate] * len(wl_data[wl_name]),
                          'fiberID': [fiberID] * len(wl_data[wl_name]),
                          'mjd': [mjd] * len(wl_data[wl_name]),
                          'dr': [dr] * len(wl_data[wl_name]),
                         'time': np.arange(len(wl_data[wl_name]))}
            df_wl = pd.concat([df_wl, pd.DataFrame(data_single)], axis = 0)

            prev_len = len_wl
            self.unique_spec_ids += [[plate, fiberID, mjd]]


        df_wl['spectra_identifier'] = df_wl[['plate', 'fiberID', 'mjd']].apply(lambda x: ','.join(x.astype(str)), axis=1)
        unique = df_wl['spectra_identifier'].unique()

        self.df = df_wl
        self.len_wl = prev_len

        return df_wl
  
    def viz_flux_wavelength(self):

        """
        Visualizes the scatter plot of wavelength against flux for all spectral objects specified by users. 

        Returns
        -------
        ax : matplotlib.axes._axes.Axes
            The plot that visualizes the wavelength against flux for all spectral objects specified by users.
        """

        if self.df is None:
            raise NotImplementedError('Please first convert the spectral data into pandas dataframe using method convert_pd().')

        data_all = self.df

        plt.figure()
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        outcome = 'wavelength: ' + self.wl_plot
        variable = 'flux'

        sns.scatterplot(data=data_all, x=variable, y=self.wl_plot, hue='spectra_identifier')

        ax.set_title('{} against {}'.format(outcome, variable), fontsize=10)
        ax.set_xlabel(variable, fontsize=8)
        ax.set_ylabel(outcome, fontsize=8)
        ax.tick_params(labelsize=6)

        return ax

    def viz_wavelength(self, plate, fiberID, mjd):

        """
            Visualizes the line plot of wavelength against flux for all spectral objects specified by users. 

            Parameters
            ----------
            plate : int
                The plate number of the SDSS spectroscopic observation.
            fiberID : int
                The fiberID number of the SDSS spectroscopic observation.
            mjd : int
                The Modified Julian Date (MJD) of the SDSS spectroscopic observation.

            Returns
            -------
            ax : matplotlib.axes._axes.Axes
                The plot that visualizes the wavelength against time for all spectral objects specified by users.
        """

        if self.df is None:
            raise NotImplementedError('Please first convert the spectral data into pandas dataframe using method convert_pd().')

        # checks on spectral identifier
        if [plate, fiberID, mjd] not in self.unique_spec_ids:
            raise ValueError('The input plate, fiberID, and mjd combination must represent one of the input spectral objects.')


        data_all = self.df

        plt.figure()
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        outcome = 'wavelength: ' + self.wl_plot
        variable = 'time'

        sns.lineplot(data=data_all, x=variable, y=self.wl_plot)

        ax.set_title('{} against {}'.format(outcome, variable), fontsize=10)
        ax.set_xlabel(variable, fontsize=8)
        ax.set_ylabel(outcome, fontsize=8)
        ax.tick_params(labelsize=6)

        return ax
