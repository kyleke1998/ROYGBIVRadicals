from astroquery.sdss import SDSS
from astropy.io import fits
import pandas as pd
from astropy import units as u
import numpy as np




class SDSSDataFetchSQL():
    """
    Create a class for fetching data that could contain multiple observations for meta data 
    from the SDSS data base using SQL.
    
    Attributes
    ----------
    sdss_base : astroquery object
        The SDSS data base object to be fetched data on.

    Methods
    -------
    __init__(self):
        Creates a class instance.
    construct_adql_query(self, sdss_table, columns, constraints):
        Writes SQL query based on user specified table name, columns, and constraints.
    fetch_data(self, adql_query, dr):
        Fetchs the data from SDSS base based on user specified SQL query and data release version.
    """
    
    def __init__(self):
        """
        Initialize a class instance for fetching data from the SDSS data base.
        
        Returns
        -------
        None
        
        """
        
        self.sdss_base = SDSS
        
    def construct_adql_query(self, sdss_table, columns, constraints):
        """
        Construct a SQL query from user specified table name, columns, and constraints in where statement.
        
        Parameters
        ----------
        sdss_table : str
            The name of table in the SDSS base to be fetched.
        columns : list
            The names of columns in the table specified of the SDSS base to be fetched. 
        constraints : str
            The statement in where statement of SQL query that specified the conditions or 
            constraints filtering on the table to be fetched.

        Returns
        -------
        query : str
            The SQL query to fetch the desirable data in SDSS data base.
        """

        if sdss_table is None:
            raise ValueError("Please specify a table name in sdss_table.")
        
        if columns is None:
            query_cols = "*"
        else:
            query_cols = ",".join(columns)

        if constraints is None:
            query = "select " + query_cols + " from " + sdss_table
        else:
            query = "select " + query_cols + " from " + sdss_table + " where " + constraints
        
        return query
    
    def fetch_data(self, adql_query, dr):
        """
        Query and output the desirable meta data as pandas data frame based on user input query.
        
        Parameters
        ----------
        adql_query : str
            A SQL query to pull data from SDSS base.
        dr : int
            The data release version. Must be DR8 or later.

        Returns
        -------
        res_df : pandas.DataFrame
            The data frame containing the meta data from SDSS base based on the user specified query.
        """
        
        if dr < 8:
            raise ValueError('The data release version must be DR8 or later.')
            
        result = self.sdss_base.query_sql(sql_query = adql_query, data_release = dr)
        res_df = result.to_pandas()
        
        return res_df
