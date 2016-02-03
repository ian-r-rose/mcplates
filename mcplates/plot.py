from itertools import cycle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import cartopy.crs as ccrs
from numba import jit

cmap_green = LinearSegmentedColormap.from_list('vphi', [ (0, '#ffffff'), (0.2, '#edf8e9'), (0.4, '#bae4b3'), (0.6, '#74c476'), (0.8, '#31a354'), (1.0, '#006d2c') ] , gamma=0.5)
cmap_green.set_bad('w', alpha=0.0)

cmap_blue = LinearSegmentedColormap.from_list('vphi', [ (0, '#ffffff'), (0.2, '#eff3ff'), (0.4, '#bdd7e7'), (0.6, '#6baed6'), (0.8, '#3182bd'), (1.0, '#08519c') ] , gamma=0.5)
cmap_blue.set_bad('w', alpha=0.0)
cmap_red = LinearSegmentedColormap.from_list('vphi', [ (0, '#ffffff'), (0.2, '#fee5d9'), (0.4, '#fcae91'), (0.6, '#fb6a4a'), (0.8, '#de2d26'), (1.0, '#a50f15') ] , gamma=0.5)
cmap_red.set_bad('w', alpha=0.0)

cmaps = cycle([cmap_red, cmap_green, cmap_blue])


def bin_trace( lon_samples, lat_samples, resolution):
    """
    Given a trace of samples in longitude and latitude, bin them 
    in latitude and longitude, and normalize the bins so that
    the integral of probability density over the sphere is one.

    The resolution keyword gives the number of divisions in latitude.
    The divisions in longitude is twice that.
    """
    lats = np.linspace( -90., 90., resolution, endpoint=False)
    lons = np.linspace( -180., 180., 2.*resolution, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons,lats)
    hist = np.zeros_like(lon_grid)

    dlon = 360. / (2.*resolution)
    dlat = 180. / resolution

    for lon, lat in zip(lon_samples, lat_samples):

        lon = np.mod(lon, 360.)
        if lon > 180.:
            lon = lon-360.
        if lat < -90. or lat > 90.:
             # Just skip invalid latitudes if they happen to arise
             continue

        lon_index = np.floor( (lon + 180.)/dlon )
        lat_index = np.floor( (lat + 90.)/dlat )
        hist[ lat_index, lon_index ] += 1.


    lat_grid += dlat/2.
    lon_grid += dlon/2.
    return lon_grid, lat_grid, hist

def density_distribution( lon_samples, lat_samples, resolution = 30 ):
    count = len(lon_samples)
    lon_grid, lat_grid, hist = bin_trace( lon_samples, lat_samples, resolution )
    return lon_grid, lat_grid, hist/count

def cumulative_density_distribution( lon_samples, lat_samples, resolution = 30 ):

    lon_grid, lat_grid, hist = bin_trace( lon_samples, lat_samples, resolution )

    # Compute the cumulative density
    hist = hist.ravel()
    i_sort = np.argsort(hist)[::-1]
    i_unsort = np.argsort(i_sort)
    hist_cumsum = hist[i_sort].cumsum()
    hist_cumsum /= hist_cumsum[-1]

    return lon_grid, lat_grid, hist_cumsum[i_unsort].reshape(lat_grid.shape)

def plot_distribution( ax, lon_samples, lat_samples, to_plot='d', resolution=30, **kwargs ):

    if 'cmap' in kwargs:
        cmap = kwargs.pop('cmap')
    else:
        cmap = next(cmaps)

    artists = []

    if 'd' in to_plot:
        lon_grid, lat_grid, density = density_distribution( lon_samples, lat_samples, resolution )
        density = ma.masked_where(density <= 0.0, density)
        a = ax.pcolormesh( lon_grid, lat_grid, density, cmap=cmap,  transform=ccrs.PlateCarree(), **kwargs)
        artists.append(a)

    if 'e' in to_plot:
        lon_grid, lat_grid, cumulative_density = cumulative_density_distribution( lon_samples, lat_samples, resolution )
        a = ax.contour( lon_grid, lat_grid, cumulative_density, levels = [0.683, 0.955], cmap=cmap, transform=ccrs.PlateCarree() )
        artists.append(a)

    if 's' in to_plot:
        a = ax.scatter( lon_samples, lat_samples, color = cmap([0.,0.5,1.])[-1], alpha = 0.1, transform = ccrs.PlateCarree(), edgecolors=None, **kwargs )
        artists.append(a)
    
    return artists
