from astroquery.sdss import SDSS
import unittest
import pytest
from unittest.mock import patch, MagicMock
import sys
import numpy as np
import copy
import scipy.interpolate as interp

sys.path.append("./src/")

from align_wavelength import align_wavelength
from fetch_data_spectra import SDSSDataFetchSpectra as SDSSDataFetch


class TestAlignWaveLength(unittest.TestCase):
    """
    Unit tests for the AlignWaveLength class.
    """

    def setUp(self):
        """
        Set up the AlignWaveLength instance and fetch data for testing.
        """

        self.spectra1 = SDSSDataFetch(17, plate=590, fiberID=204, mjd=52057)
        self.spectra2 = SDSSDataFetch(17, plate=751, fiberID=1, mjd=52251)
        self.spectra3 = SDSSDataFetch(17, plate=2340, fiberID=60, mjd=53733)
        self.spectra1.fetch()
        self.spectra2.fetch()
        self.spectra3.fetch()
        ref = self.spectra1
        obs = [self.spectra2, self.spectra3]
        self.align_spec = align_wavelength(obs, ref)

    def test_init(self):
        """
        Test initialization of AlignWaveLength instance.
        """

        ref = self.spectra1
        obs = [self.spectra2, self.spectra3]

        assert self.align_spec.obs == obs
        assert self.align_spec.ref == ref
        assert self.align_spec.ref_flux == ref.get_flux()
        assert self.align_spec.ref_wl == ref.get_wavelength()
        assert self.align_spec.wl_name is None
        assert len(self.align_spec.interp) == 0

    def test_interpolate(self):
        """
        Test interpolation of wavelength.
        """

        self.align_spec.interpolate('COADD')

        assert len(self.align_spec.interp) >= 1

    def test_align_wrong_names(self):
        """
        Test aligning with incorrect wavelength names.
        """

        ref = self.spectra3
        obs = [self.spectra2, self.spectra1]
        align_spec_wrong = align_wavelength(obs, ref)

        with pytest.raises(ValueError):
            align_spec_wrong.interpolate('Test')

    def test_align_correct(self):
        """
        Test correct alignment of spectra.
        """

        self.align_spec.interpolate('COADD')
        new_obs = self.align_spec.align()
        spectra1 = self.spectra1
        spectra2 = self.spectra2
        spectra3 = self.spectra3
        ref_spectra_flux = spectra1.get_flux()
        ref_spectra_wl = spectra1.get_wavelength()
        spectra2_flux = spectra2.get_flux()
        spectra2_wl = spectra2.get_wavelength()
        spectra3_flux = spectra3.get_flux()
        spectra3_wl = spectra3.get_wavelength()
        spectra2_new_wl = new_obs[0].get_wavelength()
        spectra3_new_wl = new_obs[1].get_wavelength()
        ref_names = list(ref_spectra_wl.keys())

        assert isinstance(new_obs, list)
        assert len(new_obs) == 2
        assert isinstance(new_obs[0], SDSSDataFetch)
        assert isinstance(new_obs[1], SDSSDataFetch)

        for wl in ref_names:
            interp_f = interp.interp1d(ref_spectra_flux[wl], ref_spectra_wl[wl], bounds_error=False)
            if wl in spectra2_wl:
                assert len(ref_spectra_wl[wl]) == len(spectra2_new_wl[wl])

            if wl in spectra3_wl:
                assert len(ref_spectra_wl[wl]) == len(spectra3_new_wl[wl])


if __name__ == "__main__":
    unittest.main()