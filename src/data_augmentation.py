import numpy as np
from scipy.signal import savgol_filter
from scipy.special import gamma


class DataAugmentation:
    def __init__(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data X must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("Input data X must be a 2D array.")
        self.X = X

    def compute_derivative(self, order=1):
        """
        Calculates the derivative of the spectrum.
        :param order: int, the order of the derivative
        :return: numpy array, the derivative of the spectral data
        """
        if not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")

        window_length = min(5, self.X.shape[1])
        if window_length % 2 == 0:
            window_length += 1

        polyorder = min(2, window_length - 1)
        return savgol_filter(
            self.X,
            window_length=window_length,
            polyorder=polyorder,
            deriv=order,
            axis=1,
        )

    def compute_frac_derivative(self, order=0.5):
        """
        Calculates the fractional derivative of the spectrum.
        :param order: float, the order of the fractional derivative
        :return: numpy array, the fractional derivative of the spectral data
        """
        if not isinstance(order, (int, float)) or order < 0:
            raise ValueError("Order must be a non-negative integer or float.")
        if order == 0:
            return self.X.copy()

        frac_derivative = np.zeros_like(self.X)
        for i in range(1, self.X.shape[1]):
            for k in range(i):
                if order - k + 1 <= 0:
                    continue
                term = (
                    (-1) ** k
                    * gamma(order + 1)
                    / (gamma(k + 1) * gamma(order - k + 1))
                    * self.X[:, i - k]
                )
                frac_derivative[:, i] += np.where(np.isfinite(term), term, 0)
        return frac_derivative

    def augment_data(self):
        """
        Augments original X matrix with its derivatives and returns the concatenated data.
        :return: numpy array, the augmented spectral data
        """
        first_derivative = self.compute_derivative(order=1)
        second_derivative = self.compute_derivative(order=2)
        fractional_derivative = self.compute_frac_derivative(order=0.5)
        return np.concatenate(
            (self.X, first_derivative, second_derivative, fractional_derivative), axis=1
        )

    @staticmethod
    def process_spectrum(spectrum, processing_function):
        """
        Processes individual spectrum data using a provided processing function.

        :param spectrum: numpy array, individual spectrum data
        :param processing_function: callable, function to process the spectrum
        :return: numpy array, processed spectrum data
        """
        if not callable(processing_function):
            raise ValueError("processing_function must be a callable.")

        return processing_function(spectrum)