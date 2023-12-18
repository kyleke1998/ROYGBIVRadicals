import unittest
from unittest import mock
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
import numpy as np
import io
import sys
sys.path.append("./src/")
from interactive_visualization import InteractiveViz


class TestInteractiveViz(unittest.TestCase):
    def setUp(self):
        self.ax = mock.MagicMock()
        self.sdss_fetcher = mock.MagicMock()
        self.iv = InteractiveViz(self.ax, self.sdss_fetcher, setup_gui=False)

    def test_init(self):
        self.assertEqual(self.iv.ax, self.ax)
        self.assertEqual(self.iv.sdss_fetcher, self.sdss_fetcher)
        self.assertIsNone(self.iv.flux)
        self.assertIsNone(self.iv.wavelength)
        self.assertIsNone(self.iv.region_selector)

    def test_onselect(self):
        # Test the onselect method with some sample data
        self.iv.wavelength = np.array([1, 2, 3, 4, 5])
        self.iv.flux = np.array([10, 20, 30, 40, 50])
        eclick = mock.MagicMock()
        eclick.xdata = 1
        erelease = mock.MagicMock()
        erelease.xdata = 4
        with unittest.mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.iv.onselect(eclick, erelease)
            output = mock_stdout.getvalue()
        expected_output = "Total Flux in the selected region (1.00 to 4.00): 40.00\n"
        self.assertEqual(output, expected_output)

    def test_zoom(self):
        self.iv.ax.get_xlim.return_value = (0, 10)
        self.iv.zoom(0.5)
        self.iv.ax.set_xlim.assert_called()

    def test_pan(self):
        self.iv.ax.get_xlim.return_value = (0, 10)
        self.iv.ax.get_ylim.return_value = (0, 20)
        self.iv.pan(2, 3)
        self.iv.ax.set_xlim.assert_called_with(2, 12)
        self.iv.ax.set_ylim.assert_called_with(3, 23)

    def test_reset_view(self):
        self.iv.wavelength = np.array([1, 2, 3, 4, 5])
        self.iv.flux = np.array([10, 20, 30, 40, 50])
        self.iv.reset_view()
        self.iv.ax.set_xlim.assert_called_with(1, 5)
        self.iv.ax.set_ylim.assert_called_with(10, 50)


if __name__ == "__main__":
    unittest.main()
