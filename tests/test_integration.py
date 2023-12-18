import unittest
from unittest.mock import MagicMock
import sys
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

sys.path.append("./src/")
from fetch_data_spectra import SDSSDataFetchSpectra
from fetch_data_meta import SDSSDataFetchSQL
from align_wavelength import align_wavelength
from preprocess import Preprocess
from machine_learning import RandomForestClassifierWrapper
from data_augmentation import DataAugmentation
from interactive_visualization import InteractiveViz
from extract_metadata import MetadataExtraction

class TestIntegration(unittest.TestCase):
    """
    Integration tests for the entire data processing and machine learning workflow.
    """

    def setUp(self):
        """
        Set up the necessary instances for testing.
        """

        self.spectra1 = SDSSDataFetchSpectra(17, plate=590, fiberID=204, mjd=52057)
        self.spectra2 = SDSSDataFetchSpectra(17, plate=751, fiberID=1, mjd=52251)
        self.spectra3 = SDSSDataFetchSpectra(17, plate=2340, fiberID=60, mjd=53733)
        self.spectra1.fetch()
        self.spectra2.fetch()
        self.spectra3.fetch()
        ref = self.spectra1
        obs = [self.spectra2, self.spectra3]
        self.align_spec = align_wavelength(obs, ref)

        self.meta = SDSSDataFetchSQL()

    def test_fetch_spectra(self):
        """
        Test fetching spectra data using SQL queries.
        """

        test_sql = self.meta.construct_adql_query('specObj', ['ra', 'dec'], 'plate = 590 and fiberID = 204 and mjd = 52057')
        test_df = self.meta.fetch_data(test_sql, 17)
        spec1_df = self.spectra1.get_meta_data()
        assert isinstance(test_df, pd.core.frame.DataFrame)
        assert isinstance(spec1_df, pd.core.frame.DataFrame)
        assert round(test_df['ra'].values[0], 3) == round(spec1_df['ra'].values[0], 3)
        assert round(test_df['dec'].values[0], 3) == round(spec1_df['dec'].values[0], 3)

        with pytest.raises(ValueError):
            self.meta.construct_adql_query(None, ['ra', 'dec'], 'plate = 590 and fiberID = 204 and mjd = 52057')

        test_sql = self.meta.construct_adql_query('specObj', ['ra', 'dec', 'plate', 'fiberID', 'mjd'], None)
        test_df = self.meta.fetch_data(test_sql, 17)
        spectra = SDSSDataFetchSpectra(17, plate=2942, fiberID=207, mjd=54521)
        spectra.fetch()
        spec_df = spectra.get_meta_data()
        assert round(test_df[(test_df['plate'] == 2942) & (test_df['fiberID'] == 207) & (test_df['mjd'] == 54521)]['ra'].values[0], 3) == round(spec_df['ra'].values[0], 3)
        assert round(test_df[(test_df['plate'] == 2942) & (test_df['fiberID'] == 207) & (test_df['mjd'] == 54521)]['dec'].values[0], 3) == round(spec_df['dec'].values[0], 3)

    def test_spectra_obj(self):
        """
        Test operations on SDSSDataFetchSpectra instances.
        """

        self.align_spec.interpolate('COADD')
        aligned = self.align_spec.align()
        for i in aligned:
            old_size_flux = i.get_flux()['COADD'].shape
            old_size_wl = i.get_wavelength()['COADD'].shape
            new_flux_error = set(np.random.normal(size=old_size_flux))
            new_wl_error = set(np.random.normal(size=old_size_wl))
            wl_name_correct = 'COADD'

            with pytest.raises(TypeError):
                i.set_flux(wl_name_correct, new_flux_error)

            with pytest.raises(TypeError):
                i.set_wavelength(wl_name_correct, new_wl_error)

            new_flux = np.random.normal(size=old_size_flux)
            new_wl = np.random.normal(size=old_size_wl)

            new_flux_wrong = np.array([1, 2, 3])
            new_wl_wrong = np.array([1, 2, 3])

            with pytest.raises(ValueError):
                i.set_flux(wl_name_correct, new_flux_wrong)

            with pytest.raises(ValueError):
                i.set_wavelength(wl_name_correct, new_wl_wrong)

            i.set_flux(wl_name_correct, new_flux)
            i.set_wavelength(wl_name_correct, new_wl)
            assert isinstance(i.flux['COADD'], np.ndarray)
            assert isinstance(i.wavelength['COADD'], np.ndarray)
            assert sum(i.flux['COADD'] != new_flux) == 0
            assert sum(i.wavelength['COADD'] != new_wl) == 0

    def test_wavelength_type(self):
        """
        Test the type of wavelength data in aligned spectra.
        """

        self.align_spec.interpolate('COADD')
        aligned = self.align_spec.align()
        for i in aligned:
            assert isinstance(i, SDSSDataFetchSpectra)

    def test_wavelength_number(self):
        """
        Test the number of aligned spectra.
        """

        self.align_spec.interpolate('COADD')
        aligned = self.align_spec.align()
        assert len(aligned) == 2

    def test_wavelength_length(self):
        """
        Test the length of wavelength data in aligned spectra.
        """

        self.align_spec.interpolate('COADD')
        aligned = self.align_spec.align()
        for i in ['COADD']:
            align_spec1 = aligned[0].get_wavelength()[i]
            align_spec2 = aligned[1].get_wavelength()[i]
            ref_spec = self.spectra1.get_wavelength()[i]

            assert len(align_spec1) == len(ref_spec)
            assert len(align_spec2) == len(ref_spec)

    def test_flux_length(self):
        """
        Test the length of flux data in aligned spectra.
        """

        self.align_spec.interpolate('COADD')
        aligned = self.align_spec.align()
        for i in ['COADD']:
            align_spec1 = aligned[0].get_flux()[i]
            align_spec2 = aligned[1].get_flux()[i]
            ref_spec = self.spectra1.get_flux()[i]

            assert len(align_spec1) == len(ref_spec)
            assert len(align_spec2) == len(ref_spec)

    def test_preprocess(self):
        """
        Test the Preprocess class for normalization, outlier removal, and redshift correction.
        """

        for i in ['COADD']:

            wl = self.spectra1.get_wavelength()[i]
            flux = self.spectra1.get_flux()[i]
            preprocess = Preprocess(wl, flux)
            normalized_data = preprocess.normalize()
            assert isinstance(normalized_data, np.ndarray)
            assert (np.all((normalized_data[1] >= 0) & (normalized_data[1] <= 1)))
            outlier_removed_data = preprocess.remove_outliers()
            assert (isinstance(outlier_removed_data, np.ndarray))
            assert ((outlier_removed_data[1] >= 0).all())
            redshift_data = preprocess.correct_redshift(5.0)
            assert isinstance(preprocess.correct_redshift(5.0), np.ndarray)
            assert (redshift_data.shape == (2, len(wl)))
            assert (5 == np.unique(np.round((wl - redshift_data[0]) / redshift_data[0], decimals=2)))

    def test_preprocess_interp(self):
        """
        Test the interpolation of flux data using the Preprocess class.
        """

        for i in ['COADD']:

            ref_wl = self.spectra1.get_wavelength()[i]
            ref_flux = self.spectra1.get_flux()[i]

            wl = self.spectra3.get_wavelength()[i]
            flux = self.spectra3.get_flux()[i]

            preprocess = Preprocess(wl, flux)

            interp_flux = preprocess.interpolate(ref_flux, ref_wl)

            assert (len(interp_flux) == len(ref_flux))
            assert (len(flux) != len(interp_flux))

    def test_machine_learning(self):
        """
        Test the machine learning workflow using RandomForestClassifierWrapper.
        """

        result = self.meta.fetch_data("""SELECT class, plate, mjd,zWarning,fiberID FROM SpecObj""", 17)
        sampled_df = result.groupby('class', group_keys=False).head(5)
        all_spectra = []
        target = []
        for i in sampled_df.iterrows():
            try:
                spectra = SDSSDataFetchSpectra(dr_release=17, plate=i[1]['plate'], fiberID=i[1]['fiberID'], mjd=i[1]['mjd'])
                spectra.fetch()
                all_spectra.append(spectra)
                target.append(i[1]['class'])
            except TypeError as e:
                print(f"Error processing row {i[0]}: {e}")
                continue  # Move to the next iteration
        interpolated_flux_list = []
        ref = all_spectra[0]
        obs = all_spectra[1:]
        for i in range(len(obs)):
            preprocess = Preprocess(obs[i].get_wavelength()['COADD'], obs[i].get_flux()['COADD'])
            new_flux = preprocess.interpolate(ref.get_flux()['COADD'], ref.get_wavelength()['COADD'])
            interpolated_flux_list.append(new_flux)
        interpolated_flux_list.append(ref.get_flux()['COADD'])
        feature_matrix = np.array(interpolated_flux_list)
        target = np.array(target)
        clf = RandomForestClassifierWrapper(n_estimators=100, max_depth=10, min_samples_split=5,
                                            min_samples_leaf=2, max_features='log2', random_state=42)
        clf.fit(feature_matrix, target)
        y_pred = clf.predict(feature_matrix)
        cm = clf.evaluate(y_pred, target)
        assert len(y_pred) == len(target)
        assert (np.array(cm).sum(axis=1).sum() == len(target))

    def test_data_augmentation(self):
        """
        Integration test for DataAugmentation with the fetched spectral data.
        """
        # Example using spectra1 data
        spectra_data = np.stack(
            [self.spectra1.get_wavelength()['COADD'], self.spectra1.get_flux()['COADD']]
        )

        # Initialize DataAugmentation with spectra data
        augmenter = DataAugmentation(spectra_data)
        self.assertIsInstance(augmenter, DataAugmentation)
        
        # Test derivative computation
        derivative = augmenter.compute_derivative(order=1)
        self.assertEqual(derivative.shape, spectra_data.shape)

        # Test fractional derivative computation
        frac_derivative = augmenter.compute_frac_derivative(order=0.5)
        self.assertEqual(frac_derivative.shape, spectra_data.shape)

        # Test data augmentation
        augmented_data = augmenter.augment_data()
        expected_columns = spectra_data.shape[1] * 4
        self.assertEqual(augmented_data.shape, (spectra_data.shape[0], expected_columns))
        self.assertIsInstance(augmented_data, np.ndarray)

    def test_metadata_extraction_integration(self):
        """
        Tests the integration of fetching data and metadata extraction.
        Integration for fetch_data_meta, fetch_data_spectra, and metadata_extraction.
        """
        # fetch a list of data objects via SQL query
        test_sql = self.meta.construct_adql_query('specObj', ['ra', 'dec', 'z', 'class', 'bestObjID', 'plate', 'mjd'], 'plate = 590 and fiberID = 204 and mjd = 52057')
        sql_df = self.meta.fetch_data(test_sql, 17)
        assert isinstance(sql_df, pd.core.frame.DataFrame)

        # fetch one data object via specifying plate, fiberID, mjd
        spec1_df = self.spectra1.get_meta_data()
        assert isinstance(spec1_df, pd.core.frame.DataFrame)

        # use resulting dataframes to create a metadata extraction instance
        metadata_extractor_sqlbased = MetadataExtraction(sql_df)
        metadata_extractor_specbased = MetadataExtraction(spec1_df)

        # test get_class functionality 
        extracted_classes = metadata_extractor_sqlbased.get_class()
        self.assertIsInstance(extracted_classes, pd.core.series.Series)
        with pytest.raises(KeyError):   # expect no class was fetched if using spectral data fetcher 
            metadata_extractor_specbased.get_class()

        # test get_coordinates functionality 
        sql_extracted_coordinates = metadata_extractor_sqlbased.get_coordinates()
        spec_extracted_coordinates = metadata_extractor_specbased.get_coordinates()
        assert isinstance(sql_extracted_coordinates, pd.core.frame.DataFrame)
        assert isinstance(spec_extracted_coordinates, pd.core.frame.DataFrame)

        # test get_identifiers functionality 
        sql_extracted_idents = metadata_extractor_sqlbased.get_identifiers(identifier_columns = ['plate','mjd'])
        spec_extracted_idents = metadata_extractor_specbased.get_identifiers(identifier_columns = ['plate','mjd'])
        assert isinstance(sql_extracted_idents, pd.core.frame.DataFrame)
        assert isinstance(spec_extracted_idents, pd.core.frame.DataFrame)

        # test get_equiv_widths functionality 
        with pytest.raises(KeyError): # did not extract equiv. widths w SQL
            metadata_extractor_sqlbased.get_equiv_widths()
        spec_extracted_eqw = metadata_extractor_specbased.get_equiv_widths()
        assert isinstance(spec_extracted_eqw, pd.core.series.Series)

        # test redshift functionality 
        sql_extracted_z = metadata_extractor_sqlbased.get_redshifts()
        spec_extracted_z = metadata_extractor_specbased.get_redshifts()
        assert isinstance(sql_extracted_z, pd.core.series.Series)
        assert isinstance(spec_extracted_z, pd.core.series.Series)

        # test custom metadata functionality 
        sql_extracted_custom = metadata_extractor_sqlbased.get_custom_metadata(variable_name = "bestObjID")
        spec_extracted_custom = metadata_extractor_specbased.get_custom_metadata(variable_name = "camcol")
        assert isinstance(sql_extracted_custom, pd.core.series.Series)
        assert isinstance(spec_extracted_custom, pd.core.series.Series)


    def test_visualization(self):
        """
        Test the visualization using the visualization module.
        """
        from visualization import spectra_visualization
        import matplotlib
        
        spec = self.spectra1
        wl = spec.get_wavelength()['COADD']
        flux = spec.get_flux()['COADD']
        preprocess = Preprocess(wl, flux)
        normalized_data = preprocess.normalize()
        new_wl = normalized_data[0]
        new_flux = normalized_data[1]
        spec.set_wavelength('COADD', new_wl)
        spec.set_flux('COADD', new_flux)
        viz = spectra_visualization([spec], 'COADD')
        df = viz.convert_pd()
        plot_fwl = viz.viz_flux_wavelength()
        plot_wl = viz.viz_wavelength(plate=590, fiberID=204, mjd=52057)
        
        assert isinstance(df, pd.core.frame.DataFrame)
        assert isinstance(plot_fwl, matplotlib.axes._axes.Axes)
        assert isinstance(plot_wl, matplotlib.axes._axes.Axes)
        
        align_spec = align_wavelength([self.spectra2, self.spectra3], spec)
        align_spec.interpolate('COADD')
        new_obs = align_spec.align()
        viz_align = spectra_visualization(new_obs, 'COADD')
        df_align = viz_align.convert_pd()
        plot_fwl_align = viz_align.viz_flux_wavelength()
        plot_wl_align = viz_align.viz_wavelength(plate=2340, fiberID=60, mjd=53733)

        assert isinstance(plot_fwl_align, matplotlib.axes._axes.Axes)
        assert isinstance(plot_wl_align, matplotlib.axes._axes.Axes)
        assert isinstance(df_align, pd.core.frame.DataFrame)
        
    def test_interactive_viz_integration(self):
        """
        Test the InteractiveViz class for its integration.
        """
        fig, ax = plt.subplots()

        mocked_fetcher = MagicMock(spec=SDSSDataFetchSpectra)
        mocked_fetcher.get_wavelength.return_value = {'COADD': np.array([4000, 5000, 6000])}
        mocked_fetcher.get_flux.return_value = {'COADD': np.array([1, 2, 3])}
        mocked_fetcher.fetch.return_value = None

        interactive_viz = InteractiveViz(ax, mocked_fetcher)
        
        interactive_viz.fetch_and_plot_data()

        mocked_fetcher.fetch.assert_called_once()
        self.assertIsNotNone(interactive_viz.wavelength)
        self.assertIsNotNone(interactive_viz.flux)
















if __name__ == "__main__":
    unittest.main()
