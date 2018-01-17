from itertools import cycle
import pkgutil
import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import cartopy.crs as ccrs

from .poles import Pole

cmap_blue = plt.get_cmap('Blues')
cmap_red = plt.get_cmap('Reds')
cmap_green = plt.get_cmap('Greens')
cmap_orange = plt.get_cmap('Oranges')
cmap_purple = plt.get_cmap('Purples')
cmap_blue.set_bad('w', alpha=0.0)
cmap_red.set_bad('w', alpha=0.0)
cmap_green.set_bad('w', alpha=0.0)
cmap_orange.set_bad('w', alpha=0.0)
cmap_purple.set_bad('w', alpha=0.0)

cmap_list = [cmap_blue, cmap_red, cmap_green, cmap_orange, cmap_purple]
cmaps = cycle(cmap_list)


continent_dictionary = { 'africa': 'af.asc',
        'australia' : 'aus.asc',
        'congo' : 'congo.asc',
        'gondwana' : 'gond.asc',
        'india' : 'ind.asc',
        'laurentia' : 'lau.asc',
        'plates' : 'plates.asc',
        'west africa' : 'waf.asc',
        'antarctica' : 'ant.asc',
        'baltica' : 'balt.asc',
        'europe' : 'eur.asc',
        'greenland' : 'grn.asc',
        'kala' : 'kala.asc',
        'north america' : 'nam.asc',
        'south america' : 'sam.asc' }


def bin_trace(lon_samples, lat_samples, resolution):
    """
    Given a trace of samples in longitude and latitude, bin them
    in latitude and longitude, and normalize the bins so that
    the integral of probability density over the sphere is one.

    The resolution keyword gives the number of divisions in latitude.
    The divisions in longitude is twice that.
    """
    lats = np.linspace(-90., 90., resolution, endpoint=True)
    lons = np.linspace(-180., 180., 2. * resolution, endpoint=True)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    hist = np.zeros_like(lon_grid)

    dlon = 360. / (2. * resolution)
    dlat = 180. / resolution

    for lon, lat in zip(lon_samples, lat_samples):

        lon = np.mod(lon, 360.)
        if lon > 180.:
            lon = lon - 360.
        if lat < -90. or lat > 90.:
            # Just skip invalid latitudes if they happen to arise
            continue

        lon_index = int(np.floor((lon + 180.) / dlon))
        lat_index = int(np.floor((lat + 90.) / dlat))
        hist[lat_index, lon_index] += 1

    lat_grid += dlat / 2.
    lon_grid += dlon / 2.
    return lon_grid, lat_grid, hist


def density_distribution(lon_samples, lat_samples, resolution=30):
    count = len(lon_samples)
    lon_grid, lat_grid, hist = bin_trace(lon_samples, lat_samples, resolution)
    return lon_grid, lat_grid, hist / count


def cumulative_density_distribution(lon_samples, lat_samples, resolution=30):

    lon_grid, lat_grid, hist = bin_trace(lon_samples, lat_samples, resolution)

    # Compute the cumulative density
    hist = hist.ravel()
    i_sort = np.argsort(hist)[::-1]
    i_unsort = np.argsort(i_sort)
    hist_cumsum = hist[i_sort].cumsum()
    hist_cumsum /= hist_cumsum[-1]

    return lon_grid, lat_grid, hist_cumsum[i_unsort].reshape(lat_grid.shape)


def plot_distribution(ax, lon_samples, lat_samples, to_plot='d', resolution=30, **kwargs):

    if 'cmap' in kwargs:
        cmap = kwargs.pop('cmap')
    else:
        cmap = next(cmaps)

    artists = []

    if 'd' in to_plot:
        lon_grid, lat_grid, density = density_distribution(
            lon_samples, lat_samples, resolution)
        density = ma.masked_where(density <= 0.05*density.max(), density)
        a = ax.pcolormesh(lon_grid, lat_grid, density, cmap=cmap,
                          transform=ccrs.PlateCarree(), **kwargs)
        artists.append(a)

    if 'e' in to_plot:
        lon_grid, lat_grid, cumulative_density = cumulative_density_distribution(
            lon_samples, lat_samples, resolution)
        a = ax.contour(lon_grid, lat_grid, cumulative_density, levels=[
                       0.683, 0.955], cmap=cmap, transform=ccrs.PlateCarree())
        artists.append(a)

    if 's' in to_plot:
        a = ax.scatter(lon_samples, lat_samples, color=cmap(
            [0., 0.5, 1.])[-1], alpha=0.1, transform=ccrs.PlateCarree(), edgecolors=None, **kwargs)
        artists.append(a)

    return artists

def plot_continent( ax, name, rotation_pole=None, angle=0.,  **kwargs):
    # Load the lat/lon file
    datastream = pkgutil.get_data('mcplates', 'data/continents/' + continent_dictionary[name])
    datalines = [line.strip()
                 for line in datastream.decode('ascii').split('\n') if line.strip()]

    # Parse the file
    lon_lat = []
    for line in datalines:
        lon,lat = float(line.split()[1]), float(line.split()[0])
        if np.abs(lon-1000.0) < 1.e-6 or np.abs( lon ) + np.abs(lat) < 1.e-6:
            lon_lat.append( [np.nan, np.nan] ) # nans break up line segments
        else:
            lon = lon if lon < 180. else lon-360.
            lon_lat.append( [lon,lat] )

    # If the user has included an Euler rotation, do that.
    if rotation_pole is not None:
        for i, ll in enumerate(lon_lat):
            pole = Pole(ll[0], ll[1], 1.0)
            pole.rotate( rotation_pole, angle )
            lon_lat[i][0] = pole.longitude
            lon_lat[i][1] = pole.latitude

    lon_lat = np.array(lon_lat)
    # Sometimes the last point messes up the plot (for reasons I don't understand).
    # Just exclude it.
    artist = ax.plot( lon_lat[:-1,0], lon_lat[:-1,1] , transform=ccrs.PlateCarree(), **kwargs)
    return artist
