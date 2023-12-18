import unittest
import unittest.mock
import numpy as np
from scipy.signal import savgol_filter
import sys
from astroquery.sdss import SDSS

sys.path.append("./src/")
from data_augmentation import DataAugmentation

class TestDataAugmentation(unittest.TestCase):
    def setUp(self):
        spectra = SDSS.get_spectra(plate=590, fiberID=204, mjd=52057)
        self.sample_data = np.stack(
            [spectra[0][1].data["loglam"], spectra[0][1].data["flux"]]
        )

    def test_initialization(self):
        augmenter = DataAugmentation(self.sample_data)
        self.assertIsInstance(augmenter, DataAugmentation)
        self.assertIsInstance(augmenter.X, np.ndarray)
        self.assertTrue(np.array_equal(augmenter.X, self.sample_data))

        with self.assertRaises(ValueError):
            DataAugmentation(42)  # Invalid input, should raise ValueError

    def test_compute_derivative_standard(self):
        augmenter = DataAugmentation(self.sample_data)
        derivative = augmenter.compute_derivative(order=1)

        expected_derivative = savgol_filter(
            self.sample_data, window_length=5, polyorder=2, deriv=1, axis=1
        )

        self.assertTrue(np.allclose(derivative, expected_derivative))

        with self.assertRaises(ValueError):
            augmenter.compute_derivative(order=-1)
        with self.assertRaises(ValueError):
            augmenter.compute_derivative(order="invalid")
        with self.assertRaises(ValueError):
            augmenter.compute_derivative(order=1.5)

    def test_compute_derivative_with_mock(self):
        with unittest.mock.patch(
            "data_augmentation.savgol_filter",
            return_value=np.ones_like(self.sample_data),
        ):
            augmenter = DataAugmentation(self.sample_data)
            derivative = augmenter.compute_derivative(order=1)

        self.assertTrue(np.array_equal(derivative, np.ones_like(self.sample_data)))

    def test_compute_frac_derivative(self):
        augmenter = DataAugmentation(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        frac_derivative = augmenter.compute_frac_derivative(order=0.5)

        expected_frac_derivative = np.array([[0, 2, 2], [0, 5, 3.5]])

        np.testing.assert_allclose(
            frac_derivative, expected_frac_derivative, rtol=1e-5, atol=1e-8
        )

        with self.assertRaises(ValueError):
            augmenter.compute_frac_derivative(order=-1)
        with self.assertRaises(ValueError):
            augmenter.compute_frac_derivative(order="invalid")

    def test_augment_data(self):
        augmenter = DataAugmentation(self.sample_data)
        augmented_data = augmenter.augment_data()

        expected_columns = self.sample_data.shape[1] * 4
        self.assertEqual(
            augmented_data.shape, (self.sample_data.shape[0], expected_columns)
        )
        self.assertIsInstance(augmented_data, np.ndarray)

    def test_data_integrity(self):
        augmenter = DataAugmentation(self.sample_data)
        augmented_data = augmenter.augment_data()
        original_data_from_augmented = augmented_data[:, : self.sample_data.shape[1]]

        self.assertTrue(np.array_equal(self.sample_data, original_data_from_augmented))
        
    def test_process_spectrum(self):
        spectrum = np.random.rand(100)
        processed = DataAugmentation.process_spectrum(spectrum, lambda x: x**2)
        np.testing.assert_array_equal(processed, spectrum**2)
        
        with self.assertRaises(ValueError):
            DataAugmentation.process_spectrum(spectrum, "not callable")


if __name__ == "__main__":
    unittest.main()
