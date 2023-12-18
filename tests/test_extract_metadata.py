import unittest
import pytest
import numpy as np 
import pandas as pd
import sys

sys.path.append("./src/")

from extract_metadata import MetadataExtraction

class TestMetadataExtraction(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment by creating a dummy metadata dataframe
        """

        dummy_data = {
            'z': [0.4, 0.35, 0.38, 0.42, 0.31],
            'ra': [11.22, 23.45, 45.67, 78.90, 98.76],
            'dec': [-45.67, -23.45, 12.34, 34.56, 67.89],
            'plate': [101, 102, 103, 104, 105],
            'fiberID': [201, 202, 203, 204, 205],
            'mjd': [58127, 58128, 58129, 58130, 58131],
            'specObjID': [123456, 234567, 345678, 456789, 567890],
            'run': [910, 911, 912, 913, 914],
            'camcol': [1, 2, 3, 4, 5],
            'field': [101, 102, 103, 104, 105],
            'class': ['galaxy','galaxy','galaxy','galaxy','galaxy'],
            'equiv_widths': [
                [{'line': 'H-alpha', 'value': 10.0}, {'line': 'H-beta', 'value': 8.0}],
                [{'line': 'H-alpha', 'value': 12.0}, {'line': 'H-beta', 'value': 7.5}],
                [{'line': 'H-alpha', 'value': 11.0}, {'line': 'H-beta', 'value': 9.0}],
                [{'line': 'H-alpha', 'value': 9.0}, {'line': 'H-beta', 'value': 9.5}],
                [{'line': 'H-alpha', 'value': 1.0}, {'line': 'H-beta', 'value': 2.0}]
            ]
        }

        self.dummy_df = pd.DataFrame(dummy_data)
        self.metadata_extractor = MetadataExtraction(data = self.dummy_df)

    def test_init_nonexist_df(self): 
        """
        Test the handling of a metadata dataframe that has not been fetched. 

        Asserts: 
            - Raises a ValueError when a nonexistent metadata column name is provided.
        """
        non_dataframe_input = "not a dataframe"

        with self.assertRaises(ValueError): 
            metadata = MetadataExtraction(data = non_dataframe_input)

    def test_init_with_df(self):
        """
        Test the handling of initializing metadata extraction with a pandas dataframe, as expected
        """
        try: 
            metadata_extractor_setup = MetadataExtraction(data = self.dummy_df)
        except ValueError: 
            self.fail("ValueError should not be raised when providing a dataframe object for initialization")

    def test_get_class_valid(self):
        """
        Test extracting the 'class' column
        """
        class_column = self.metadata_extractor.get_class()
        expected_class_column = pd.Series(['galaxy','galaxy','galaxy','galaxy','galaxy'], name='class')
        pd.testing.assert_series_equal(class_column, expected_class_column)

    def test_get_class_missing_class_column(self):
        """
        Test handling when the 'class' column is missing
        """
        # Remove the 'class' column
        dummy_df_copy = self.dummy_df.copy()
        del dummy_df_copy['class']

        no_class_extractor = MetadataExtraction(dummy_df_copy)

        with self.assertRaises(KeyError):
            no_class_extractor.get_class()

    def test_get_class_empty_dataframe(self):
        """
        Test handling an empty DataFrame
        """
        empty_df = pd.DataFrame()
        e_extractor = MetadataExtraction(empty_df)

        with self.assertRaises(KeyError):
            e_extractor.get_class()

    def test_get_coordinates_valid(self):
        """
        Test retrieving 'ra' and 'dec' columns from data
        """
        coordinates_df = self.metadata_extractor.get_coordinates()

        # correct columns
        expected_columns = ['ra', 'dec']
        self.assertListEqual(list(coordinates_df.columns), expected_columns)

        # Check if the values in the 'ra' and 'dec' columns are correct
        expected_values = {'ra': [11.22, 23.45, 45.67, 78.90, 98.76],
                            'dec': [-45.67, -23.45, 12.34, 34.56, 67.89]}
        pd.testing.assert_frame_equal(coordinates_df, pd.DataFrame(expected_values))

    def test_get_coordinates_missing_ra_column(self):
        """
        Test handling when the 'ra' column is missing.
        """
        # Remove the 'ra' column
        dummy_df_copy = self.dummy_df.copy()
        del dummy_df_copy['ra']

        no_ra_extractor = MetadataExtraction(dummy_df_copy)

        with self.assertRaises(KeyError):
            no_ra_extractor.get_coordinates()

    def test_get_coordinates_missing_dec_column(self):
        """
        Test handling when the 'dec' column is missing.
        """
        # Remove the 'dec' column
        dummy_df_copy = self.dummy_df.copy()
        del dummy_df_copy['dec']

        no_dec_extractor = MetadataExtraction(dummy_df_copy)

        with self.assertRaises(KeyError):
            no_dec_extractor.get_coordinates()

    def test_get_coordinates_empty_dataframe(self):
        """
        Test handling an empty DataFrame.
        """
        empty_df = pd.DataFrame()

        e_extractor = MetadataExtraction(empty_df)

        with self.assertRaises(ValueError):
            e_extractor.get_coordinates()


    def test_get_identifiers_valid(self):
        """
        Test retrieving all possible identifiers from DataFrame
        """
        iden_df = self.metadata_extractor.get_identifiers(['plate', 'fiberID', 'mjd', 'specObjID', 'run', 'camcol', 'field'])
        
        # Check if the returned DataFrame has the correct shape
        self.assertEqual(iden_df.shape, (5, 7))

        # Check if the returned DataFrame has the correct columns
        expected_columns = ['plate', 'fiberID', 'mjd', 'specObjID', 'run', 'camcol', 'field']
        self.assertListEqual(list(iden_df.columns), expected_columns)

        # Check if the values in the columns are correct
        expected_values = {'plate': [101, 102, 103, 104, 105],
                            'fiberID': [201, 202, 203, 204, 205],
                            'mjd': [58127, 58128, 58129, 58130, 58131],
                            'specObjID': [123456, 234567, 345678, 456789, 567890],
                            'run': [910, 911, 912, 913, 914],
                            'camcol': [1, 2, 3, 4, 5],
                            'field': [101, 102, 103, 104, 105]}
        pd.testing.assert_frame_equal(iden_df, pd.DataFrame(expected_values))

    def test_get_identifiers_subset(self):
        """
        Test retrieving a select list of identifiers
        """
        iden_df = self.metadata_extractor.get_identifiers(['fiberID', 'specObjID', 'camcol'])

        # Check if the returned DataFrame has the correct columns
        expected_columns = ['fiberID', 'specObjID', 'camcol']
        self.assertListEqual(list(iden_df.columns), expected_columns)

        # Check if the values in the specific columns are correct
        expected_values = {'fiberID': [201, 202, 203, 204, 205],
                            'specObjID': [123456, 234567, 345678, 456789, 567890],
                            'camcol': [1, 2, 3, 4, 5]}
        pd.testing.assert_frame_equal(iden_df, pd.DataFrame(expected_values))

    def test_get_identifiers_empty_dataframe(self):
        """
        Test handling an empty DataFrame.
        """
        empty_df = pd.DataFrame()
        e_extractor = MetadataExtraction(empty_df)

        with self.assertRaises(KeyError):
            e_extractor.get_identifiers(['plate', 'fiberID'])

    def test_get_equiv_widths_valid(self):
        """
        Test retrival of equivalent widths column
        """
        result = self.metadata_extractor.get_equiv_widths()

        pd.testing.assert_series_equal(result, self.dummy_df.equiv_widths)



    def test_get_equiv_widths_empty_dataframe(self):
        """
        Test handling an empty DataFrame.
        """
        empty_df = pd.DataFrame()
        e_extractor = MetadataExtraction(empty_df)

        with self.assertRaises(KeyError):
            e_extractor.get_equiv_widths()

    def test_get_equiv_widths_missing_column(self):
        """
        Test handling when "equiv_widths" column is missing
        """
        dummy_df_copy = self.dummy_df
        df_missing_column = dummy_df_copy.drop(columns='equiv_widths')

        noeq_extractor = MetadataExtraction(df_missing_column)
        with self.assertRaises(KeyError):
            noeq_extractor.get_equiv_widths()

    def test_get_redshift_valid(self): 
        """
        Test retrival of redshift values 'z' column
        """
        result = self.metadata_extractor.get_redshifts()
        expected_redshift_column = pd.Series([0.4, 0.35, 0.38, 0.42, 0.31], name='z')
        pd.testing.assert_series_equal(result, expected_redshift_column)

    def test_get_redshift_missing_column(self): 
        """
        Test handling when "z" column is missing
        """
        dummy_df_copy = self.dummy_df
        df_missing_column = dummy_df_copy.drop(columns='z')

        noz_extractor = MetadataExtraction(df_missing_column)
        with self.assertRaises(KeyError):
            noz_extractor.get_redshifts()

    def test_get_redshift_empty_dataframe(self): 
        """
        Test handling an empty DataFrame.
        """
        empty_df = pd.DataFrame()
        e_extractor = MetadataExtraction(empty_df)

        with self.assertRaises(KeyError):
            e_extractor.get_redshifts()

    def test_get_custom_metadata_valid(self):
        """
        Test retrieval of column based on valid col name requested
        """
        result = self.metadata_extractor.get_custom_metadata(variable_name = 'z')
        expected = pd.Series([0.4, 0.35, 0.38, 0.42, 0.31], name='z')
        pd.testing.assert_series_equal(result, expected)

    def test_get_custom_metadata_invalid_name(self): 
        """
        Test retrieval of column based on invalid col name requested
        """
        with self.assertRaises(KeyError): 
            self.metadata_extractor.get_custom_metadata(variable_name = 'not a name')





if __name__ == "__main__":
    unittest.main()
