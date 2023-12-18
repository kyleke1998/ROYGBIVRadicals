from astroquery.sdss import SDSS
import unittest
from unittest.mock import patch, MagicMock
import sys
import numpy as np
import pytest
import pandas as pd

sys.path.append("./src/")
from fetch_data_meta import SDSSDataFetchSQL


class TestSDSSDataFetch(unittest.TestCase):
    """
    Unit tests for the SDSSDataFetchSQL class.

    Methods:
    - test_init: Test the initialization of the SDSSDataFetchSQL instance.
    - test_construct_adql_query_normal: Test constructing an ADQL query with normal inputs.
    - test_construct_adql_query_no_columns: Test constructing an ADQL query with no specified columns.
    - test_construct_adql_query_no_constraints: Test constructing an ADQL query with no constraints.
    - test_construct_adql_query_raises_value_error: Test constructing an ADQL query with invalid inputs.
    - test_fetch_data: Test fetching data using an SQL query.
    - test_dr_wrong: Test fetching data with an incorrect data release version.
    """

    def setUp(self):
        """
        Set up the SDSSDataFetchSQL instance for testing.
        """
        self.fetcher = SDSSDataFetchSQL()

    def test_init(self):
        """
        Test the initialization of the SDSSDataFetchSQL instance.
        """
        self.assertEqual(self.fetcher.sdss_base, SDSS)

    def test_construct_adql_query_normal(self):
        """
        Test constructing an ADQL query with normal inputs.
        """
        query = self.fetcher.construct_adql_query(
            "stars", ["name", "color"], "color = 'red'"
        )
        self.assertEqual(query, "select name,color from stars where color = 'red'")

    def test_construct_adql_query_no_columns(self):
        """
        Test constructing an ADQL query with no specified columns.
        """
        query = self.fetcher.construct_adql_query("planets", None, None)
        self.assertEqual(query, "select * from planets")

    def test_construct_adql_query_no_constraints(self):
        """
        Test constructing an ADQL query with no constraints.
        """
        query = self.fetcher.construct_adql_query("galaxies", ["name"], None)
        self.assertEqual(query, "select name from galaxies")

    def test_construct_adql_query_raises_value_error(self):
        """
        Test constructing an ADQL query with invalid inputs.
        """
        with self.assertRaises(ValueError):
            self.fetcher.construct_adql_query(None, ["name"], None)

    def test_fetch_data(self):
        """
        Test fetching data using an SQL query.
        """
        test_sql = "select top 10 z, ra, dec, bestObjID from specObj where class = 'galaxy' and z > 0.3 and zWarning = 0"
        df = self.fetcher.fetch_data(test_sql, 17)
        assert isinstance(df, pd.core.frame.DataFrame)

    def test_dr_wrong(self):
        """
        Test fetching data with an incorrect data release version.
        """
        test_sql = "select top 10 z, ra, dec, bestObjID from specObj where class = 'galaxy' and z > 0.3 and zWarning = 0"
        with pytest.raises(ValueError) as excinfo:
            self.fetcher.fetch_data(test_sql, 6)


if __name__ == "__main__":
    unittest.main()
