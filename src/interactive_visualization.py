import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from fetch_data_spectra import SDSSDataFetchSpectra

class InteractiveViz:
    """
    Interactive visualization of SDSS spectra.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes instance
    sdss_fetcher : SDSSDataFetchSpectra instance
    setup_gui : bool
    
    Methods
    -------
    fetch_and_plot_data()
        Fetch and plot data from SDSS.
    onselect(eclick, erelease)
        Callback function for the region selector.
    update_plot()
        Update the plot.
    update_region_selector()
        Update the region selector.
    zoom(scale_factor)
        Zoom in or out.
    pan(dx, dy)
        Pan the plot.
    init_region_selector()
        Initialize the region selector.
    reset_view()
        Reset the view.
    """

    def __init__(self, ax, sdss_fetcher, setup_gui=True):
        """Initialize the interactive visualization.
        
        :param ax : matplotlib.axes.Axes instance
            The axes to plot the data on.
        :param sdss_fetcher : SDSSDataFetchSpectra instance
        :param setup_gui : bool
            Whether to setup the GUI.
        """
        self.ax = ax
        self.sdss_fetcher = sdss_fetcher
        self.flux = None
        self.wavelength = None
        self.region_selector = None
        if setup_gui:
            self.init_region_selector()

    def init_region_selector(self):
        """Initialize the region selector."""
        self.region_selector = RectangleSelector(
            self.ax, self.onselect,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True)

    def fetch_and_plot_data(self):
        """Fetch and plot data from SDSS."""
        self.sdss_fetcher.fetch()
        self.wavelength = self.sdss_fetcher.get_wavelength()['COADD']
        self.flux = self.sdss_fetcher.get_flux()['COADD']
        if self.wavelength is None or self.flux is None:
            return  # Handle the case where data is not available
        self.update_plot()

    def onselect(self, eclick, erelease):
        """Callback function for the region selector.
        
        :param eclick : matplotlib.widgets.RectangleSelector event
        :param erelease : matplotlib.widgets.RectangleSelector event
        """
        if self.wavelength is None or self.flux is None:
            return
        
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        start_index = np.searchsorted(self.wavelength, x1)
        end_index = np.searchsorted(self.wavelength, x2)
        selected_flux = self.flux[start_index:end_index]
        total_flux = np.trapz(selected_flux, self.wavelength[start_index:end_index])
        print(f"Total Flux in the selected region ({x1:.2f} to {x2:.2f}): {total_flux:.2f}")

    def zoom(self, scale_factor):
        """Zoom the plot by scale_factor.
        
        :param scale_factor : float
        """
        xlim = self.ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        x_mid = np.mean(xlim)
        self.ax.set_xlim(x_mid - x_range * scale_factor / 2, x_mid + x_range * scale_factor / 2)
        self.ax.figure.canvas.draw_idle()

    def pan(self, dx, dy):
        """Pan the plot by dx and dy.
        
        Parameters
        ----------
        :param dx : float
        :param dy : float
        """
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        self.ax.figure.canvas.draw_idle()

    def reset_view(self):
        """Reset the view to the original plot."""
        self.ax.set_xlim(np.min(self.wavelength), np.max(self.wavelength))
        self.ax.set_ylim(np.min(self.flux), np.max(self.flux))
        self.ax.figure.canvas.draw_idle()

    def update_plot(self):
        """Update the plot."""
        self.ax.clear()
        if self.wavelength is not None and self.flux is not None:
            self.ax.plot(self.wavelength, self.flux, label="Spectral Data")
            self.ax.legend()
        self.init_region_selector()
        self.ax.figure.canvas.draw()
        