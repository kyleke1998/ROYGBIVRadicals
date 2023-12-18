import pandas as pd 
import numpy as np

class MetadataExtraction():

    def __init__(self, data):
        """
        Initialize MetadataExtraction with a pandas DataFrame.
        Assumes that you have fetched meta data using SDSSDataFetchSQL.fetch_data method in the fetch_data_metamodule.
        That is the data you input to this module. 

        Parameters:
            - data: Pandas DataFrame containing metadata.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        self.data = data 

    def get_class(self):
        """
        Retrieves the 'class' column from the data

        Returns:
            pd.Series: The extracted 'class' column.
        """
        try:
            class_column = self.data['class']
            return class_column
        except KeyError:
            raise KeyError("The 'class' column is missing in the provided data.")


    def get_coordinates(self):
        """
        Retrieves 'ra' and 'dec' columns from the data

        Returns:
            pd.DataFrame: A DataFrame containing 'ra' and 'dec' columns.
        """
        if self.data.empty:
            raise ValueError("DataFrame is empty.")
        try:
            coordinates_df = self.data[['ra', 'dec']]
            return coordinates_df
        except KeyError as e:
            missing_column = str(e).strip("'")
            raise KeyError(f"The '{missing_column}' column is missing in the provided data.")
        except ValueError:
            raise ValueError("Cannot retrieve coordinates from an empty DataFrame.")


    def get_identifiers(self, identifier_columns):
        """
        Retrieves specified identifiers from the data

        Args:
            identifier_columns (list): List of columns to retrieve as identifiers.

        Returns:
            pd.DataFrame: A DataFrame containing the specified identifiers.

        Notes:
        The valid identifiers are: plate, fiberID, mjd, specObjID, run, camcol, field.

        """
        valid_identifiers = ['plate', 'fiberID', 'mjd', 'specObjID', 'run', 'camcol', 'field']

        # Check if all specified identifiers are valid
        invalid_identifiers = set(identifier_columns) - set(valid_identifiers)
        if invalid_identifiers:
            raise ValueError(f"Invalid identifier(s) specified: {', '.join(invalid_identifiers)}. "
                             f"Valid identifiers are: {', '.join(valid_identifiers)}.")

        try:
            identifiers_df = self.data[identifier_columns]
            if identifiers_df.empty:
                raise ValueError("Cannot retrieve identifiers from an empty DataFrame.")
            return identifiers_df
        except KeyError as e:
            missing_column = str(e).strip("'")
            raise KeyError(f"The '{missing_column}' column is missing in the provided data.")


    def get_equiv_widths(self):
        """
        Retrieves the 'equiv_widths' column from the data

        Returns:
            series: A series of equivalent width information for each data object

        """
        try:
        # Check if the 'equiv_widths' column is present in the DataFrame
            equiv_widths_series = self.data['equiv_widths']
        except KeyError:
            raise KeyError("The 'equiv_widths' column is missing in the provided data.")

        return equiv_widths_series

    def get_redshifts(self):
        """
        Retrieves the 'z' column (redshifts) from the data

        Returns:
            pd.Series: The extracted 'z' column.

        """
        try:
            redshift_column = self.data['z']
            return redshift_column
        except KeyError:
            raise KeyError("The redshift 'z' column is missing in the provided data.")

    def get_custom_metadata(self, variable_name: str):
        """
        Retrieves a specific column input by the user from the data

        Returns:
            pd.Series: The extracted metadata column.

        """
        try:
            column = self.data[variable_name]
            return column
        except KeyError:
            raise KeyError(f"The user input '{variable_name}'column is missing in the provided data.")
       

