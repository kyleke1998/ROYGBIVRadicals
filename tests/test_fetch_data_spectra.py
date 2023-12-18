from astroquery.sdss import SDSS
import unittest
import pytest
from unittest.mock import patch, MagicMock
import sys
import numpy as np

sys.path.append("./src/")

from fetch_data_spectra import SDSSDataFetchSpectra

class TestSDSSDataFetch(unittest.TestCase):
    """
    Unit tests for the SDSSDataFetchSpectra class.
    """

    def setUp(self):
        """
        Set up the SDSSDataFetchSpectra instance for testing.
        """

        self.fetcher = SDSSDataFetchSpectra(dr_release=17, plate=590, fiberID=204, mjd=52057)

    def test_dr(self):
        """
        Test handling incorrect data release version during initialization.
        """

        dr = 5
        with pytest.raises(ValueError):
            SDSSDataFetchSpectra(dr_release=dr, plate=590, fiberID=204, mjd=52057)

    def test_empty_wl(self):
        """
        Test raising NotImplementedError for get_wavelength when no data is fetched.
        """

        with pytest.raises(NotImplementedError):
            self.fetcher.get_wavelength()

    def test_empty_flux(self):
        """
        Test raising NotImplementedError for get_flux when no data is fetched.
        """

        with pytest.raises(NotImplementedError):
            self.fetcher.get_flux()

    def test_empty_meta(self):
        """
        Test raising NotImplementedError for get_meta_data when no data is fetched.
        """

        with pytest.raises(NotImplementedError):
            self.fetcher.get_meta_data()

    def test_get_wl(self):
        """
        Test retrieving wavelength data after fetching.
        """

        self.fetcher.fetch()
        spec = SDSS.get_spectra(plate=590, fiberID=204, mjd=52057, data_release=17)
        wl_names = ['COADD', 'B1-00010112-00010110-00010111', 'B1-00010113-00010110-00010111', 'B1-00010114-00010110-00010111', 'R1-00010112-00010110-00010111', 'R1-00010113-00010110-00010111', 'R1-00010114-00010110-00010111']
        wl_idx = [1, 4, 5, 6, 7, 8, 9]
        wl_data = self.fetcher.get_wavelength()

        for idx, wl in enumerate(wl_names):
            assert len(wl_data[wl]) == len(spec[0][wl_idx[idx]].data['loglam'])
            assert wl == list(wl_data.keys())[idx]

        assert isinstance(self.fetcher.wavelength, dict)

    def test_get_flux(self):
        """
        Test retrieving flux data after fetching.
        """

        self.fetcher.fetch()
        spec = SDSS.get_spectra(plate=590, fiberID=204, mjd=52057, data_release=17)
        wl_names = ['COADD', 'B1-00010112-00010110-00010111', 'B1-00010113-00010110-00010111', 'B1-00010114-00010110-00010111', 'R1-00010112-00010110-00010111', 'R1-00010113-00010110-00010111', 'R1-00010114-00010110-00010111']
        wl_idx = [1, 4, 5, 6, 7, 8, 9]
        flux_data = self.fetcher.get_flux()

        for idx, wl in enumerate(wl_names):
            assert len(flux_data[wl]) == len(spec[0][wl_idx[idx]].data['flux'])
            assert wl == list(flux_data.keys())[idx]

        assert isinstance(self.fetcher.flux, dict)

    def test_fetch(self):
        """
        Test fetching data and validating wavelength and flux data.
        """

        spec = SDSS.get_spectra(plate=590, fiberID=204, mjd=52057, data_release=17)
        meta = SDSS.query_specobj(plate=590, fiberID=204, mjd=52057, data_release=17)
        self.fetcher.fetch()

        wl_names = ['COADD', 'B1-00010112-00010110-00010111', 'B1-00010113-00010110-00010111', 'B1-00010114-00010110-00010111', 'R1-00010112-00010110-00010111', 'R1-00010113-00010110-00010111', 'R1-00010114-00010110-00010111']
        wl_idx = [1, 4, 5, 6, 7, 8, 9]
        flux_data = self.fetcher.flux
        wl_data = self.fetcher.wavelength

        for idx, wl in enumerate(wl_names):
            assert len(wl_data[wl]) == len(spec[0][wl_idx[idx]].data['loglam'])
            assert len(flux_data[wl]) == len(spec[0][wl_idx[idx]].data['flux'])
            assert wl == list(flux_data.keys())[idx]
            assert wl == list(wl_data.keys())[idx]

        assert isinstance(self.fetcher.wavelength, dict)
        assert isinstance(self.fetcher.flux, dict)
        assert isinstance(self.fetcher.meta.equiv_widths[0], dict)

    def test_set_wl_length_wrong(self):
        """
        Test setting wavelength with incorrect length.
        """

        self.fetcher.fetch()
        new_wl = np.array([1, 2, 3])
        wl_name = 'COADD'
        with pytest.raises(ValueError):
            self.fetcher.set_wavelength(wl_name, new_wl)

    def test_set_flux_length_wrong(self):
        """
        Test setting flux with incorrect length.
        """

        self.fetcher.fetch()
        new_flux = np.array([1, 2, 3])
        wl_name = 'COADD'
        with pytest.raises(ValueError):
            self.fetcher.set_flux(wl_name, new_flux)

    def test_set_wl_length_correct(self):
        """
        Test setting wavelength with correct length.
        """

        self.fetcher.fetch()
        old_size = self.fetcher.get_wavelength()['COADD'].shape
        new_wl = np.random.normal(size=old_size)
        wl_name = 'COADD'
        self.fetcher.set_wavelength(wl_name, new_wl)
        assert isinstance(self.fetcher.wavelength[wl_name], np.ndarray)
        assert sum(self.fetcher.wavelength[wl_name] != new_wl) == 0

    def test_set_flux_length_correct(self):
        """
        Test setting flux with correct length.
        """

        self.fetcher.fetch()
        old_size = self.fetcher.get_flux()['COADD'].shape
        new_flux = np.random.normal(size=old_size)
        wl_name = 'COADD'
        self.fetcher.set_flux(wl_name, new_flux)
        assert isinstance(self.fetcher.flux[wl_name], np.ndarray)
        assert sum(self.fetcher.flux[wl_name] != new_flux) == 0

    def test_set_wl_name(self):
        """
        Test setting wavelength with a new name.
        """

        self.fetcher.fetch()
        old_size = self.fetcher.get_wavelength()['COADD'].shape
        new_wl = np.random.normal(size=old_size)
        wl_name = 'Test'
        self.fetcher.set_wavelength(wl_name, new_wl)
        new_fetch_wl = list(self.fetcher.get_wavelength().keys())
        assert wl_name in new_fetch_wl

    def test_set_flux_name(self):
        """
        Test setting flux with a new name.
        """

        self.fetcher.fetch()
        old_size = self.fetcher.get_flux()['COADD'].shape
        new_flux = np.random.normal(size=old_size)
        wl_name = 'Test'
        self.fetcher.set_flux(wl_name, new_flux)
        new_fetch_wl = list(self.fetcher.get_flux().keys())
        assert wl_name in new_fetch_wl

    def test_flux_nan(self):
        """
        Test ensuring no NaN values in fetched flux data.
        """

        spectra2 = SDSSDataFetchSpectra(17, plate=751, fiberID=1, mjd=52251)
        spectra2.fetch()
        flux_data = spectra2.get_flux()
        wl_data = spectra2.get_wavelength()
        wl_names = list(wl_data.keys())

        for wl in wl_names:
            assert not np.isnan(flux_data[wl]).any()
            assert not np.isnan(wl_data[wl]).any()

    def test_set_flux_type(self):
        """
        Test setting flux with an incorrect data type.
        """

        self.fetcher.fetch()
        old_size = self.fetcher.get_flux()['COADD'].shape
        new_flux = set(np.random.normal(size=old_size))
        wl_name = 'COADD'
        with pytest.raises(TypeError):
            self.fetcher.set_flux(wl_name, new_flux)

    def test_set_wavelength_type(self):
        """
        Test setting wavelength with an incorrect data type.
        """

        self.fetcher.fetch()
        old_size = self.fetcher.get_wavelength()['COADD'].shape
        new_wl = set(np.random.normal(size=old_size))
        wl_name = 'COADD'
        with pytest.raises(TypeError):
            self.fetcher.set_wavelength(wl_name, new_wl)



if __name__ == "__main__":
    unittest.main()