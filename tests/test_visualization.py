from astroquery.sdss import SDSS
import unittest
import pytest
from unittest.mock import patch, MagicMock
import sys
import numpy as np
import copy
import scipy.interpolate as interp
import matplotlib
import pandas as pd

sys.path.append("./src/")

from align_wavelength import align_wavelength
from fetch_data_meta import SDSSDataFetchSQL
from fetch_data_spectra import SDSSDataFetchSpectra as SDSSDataFetch
from visualization import spectra_visualization


class TestSpectraVisualization(unittest.TestCase):
    
    def setUp(self):
        self.spectra1 = SDSSDataFetch(17, plate=590, fiberID=204, mjd=52057)
        self.spectra2 = SDSSDataFetch(17, plate=751, fiberID = 1, mjd=52251)
        self.spectra3 = SDSSDataFetch(17, plate=2340, fiberID=60, mjd=53733)
        self.spectra1.fetch()
        self.spectra2.fetch()
        self.spectra3.fetch()
        ref = self.spectra1
        obs = [self.spectra2, self.spectra3]
        align_spec = align_wavelength(obs, ref)
        align_spec.interpolate('COADD')
        self.new_obs = align_spec.align()
        
        obs1 = [self.spectra3]
        align_spec1 = align_wavelength(obs1, ref)
        align_spec1.interpolate('COADD')
        self.new_obs1 = align_spec1.align()
    
    def test_init(self):
        
        viz = spectra_visualization(self.new_obs, 'COADD')
        
        assert viz.data == self.new_obs
        assert viz.df is None
        assert viz.len_wl is None
        assert viz.wl_plot == 'COADD'
        assert len(viz.unique_spec_ids) == 0
#         assert len(viz.unique_fiberID) == 0
#         assert len(viz.unique_mjd) == 0
    
    def test_convert_pd(self):
        
        for wl in ['COADD']:
            viz = spectra_visualization(self.new_obs1, wl)
            df = viz.convert_pd()
            nobject = len(self.new_obs1)
            wl_size = len(self.new_obs1[0].get_wavelength()[wl])


            assert viz.len_wl == wl_size
            assert viz.df is not None
            assert len(viz.unique_spec_ids) == nobject
            assert isinstance(viz.df, pd.core.frame.DataFrame)
            assert viz.df.shape == (nobject * wl_size, 8)
            assert list(viz.df.columns) == [wl, 'flux', 'plate', 'fiberID', 'mjd', 'dr', 'time', 'spectra_identifier']
    
    def test_convert_pd_wrong(self):
        for wl in ['R1-2', 'R1-3', 'R1-4']:
            viz = spectra_visualization(self.new_obs, wl)
            
            with pytest.raises(ValueError) as excinfo:  
                df = viz.convert_pd()
    
    def test_viz_flux_wl(self):
        
        for wl in ['COADD']:
            viz = spectra_visualization(self.new_obs, wl)
            df = viz.convert_pd()
            plot_fwl = viz.viz_flux_wavelength()
            plot_wl = viz.viz_wavelength(plate=2340, fiberID=60, mjd=53733)

            assert isinstance(plot_fwl, matplotlib.axes._axes.Axes)
            assert isinstance(plot_wl, matplotlib.axes._axes.Axes)
    
    def test_viz_no_df(self):
        
        viz = spectra_visualization(self.new_obs, 'COADD')
        
        with pytest.raises(NotImplementedError) as excinfo:  
            df = viz.viz_flux_wavelength()
        
        with pytest.raises(NotImplementedError) as excinfo:  
            df = viz.viz_wavelength(plate=2340, fiberID=60, mjd=53733)
    
    def test_viz_wrong_id(self):
        
        viz = spectra_visualization(self.new_obs, 'COADD')
        df = viz.convert_pd()
        plt_plate = 591
        plt_fiberID = 2
        plt_mjd = 51000
        
        with pytest.raises(ValueError) as excinfo:  
            df = viz.viz_wavelength(plt_plate, 60, 53733)
        
        with pytest.raises(ValueError) as excinfo:  
            df = viz.viz_wavelength(2340, plt_fiberID, 53733)
        
        with pytest.raises(ValueError) as excinfo:  
            df = viz.viz_wavelength(2340, 60, plt_mjd)
        
        with pytest.raises(ValueError) as excinfo:  
            df = viz.viz_wavelength(590, 60, 53733)
        
        with pytest.raises(ValueError) as excinfo:  
            df = viz.viz_wavelength(2340, 1, 53733)
        
        
        
        