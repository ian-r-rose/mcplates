import copy

import numpy as np
from scipy.constants import Julian_year

import matplotlib.pyplot as plt
import matplotlib.path
import matplotlib.patches

import theano.tensor as tt

import cartopy.crs as ccrs

from . import rotations

class Pole(object):
    """
    Class representing a pole on the globe:
    essentially a 3-vector with some additional
    properties and operations.
    """
    def __init__(self, longitude, latitude, norm, angular_error=None):
        """
        Initialize the pole with lon, lat, and norm.
        """
        self._pole = rotations.spherical_to_cartesian(longitude, latitude, norm)
        #self._pole_numpy = rotations.spherical_to_cartesian_numpy(longitude,latitude,norm)
        self._angular_error = angular_error

    @property
    def longitude(self):
        return tt.arctan2(self._pole[1], self._pole[0] )*rotations.r2d

    @property
    def latitude(self):
        return 90. - tt.arccos(self._pole[2]/self.norm)*rotations.r2d

    @property
    def colatitude(self):
        return tt.arccos(self._pole[2]/self.norm)*rotations.r2d

    @property
    def norm(self):
        return tt.sqrt(self._pole[0]*self._pole[0] + self._pole[1]*self._pole[1] + self._pole[2]*self._pole[2])

    @property
    def angular_error(self):
        return self._angular_error

    def copy(self):
        return copy.deepcopy(self)

    def rotate(self, pole, angle):
        # The idea is to rotate the pole so that the Euler pole is
        # at the pole of the coordinate system, then perform the
        # requested rotation, then restore things to the original
        # orientation 
        p = tt.as_tensor_variable(self._pole)
        lon,lat,norm = rotations.cartesian_to_spherical(pole._pole)
        colat = 90.-lat
        p = rotations.rotate_z(p, -lon[0]*rotations.d2r)
        p = rotations.rotate_y(p, -colat[0]*rotations.d2r)
        p = rotations.rotate_z(p, angle*rotations.d2r)
        p = rotations.rotate_y(p, colat[0]*rotations.d2r)
        self._pole = rotations.rotate_z(p, lon[0]*rotations.d2r)

    def plot(self, axes, **kwargs):
        artists = []
        plon,plat,pnorm = rotations.cartesian_to_spherical_numpy(self._pole_numpy)
        if self._angular_error is not None:
            lons = np.linspace(0., 360., 361.)
            lats = np.ones_like(lons)*(90.-self.angular_error)
            norms = np.ones_like(lons)
            vecs = rotations.spherical_to_cartesian_numpy(lons,lats,norms)
            rotation_matrix = rotations.construct_euler_rotation_matrix_numpy( 0., (90.-plat[0])*rotations.d2r, plon[0]*rotations.d2r )
            rotated_vecs = np.dot(rotation_matrix, vecs)
            lons,lats,norms = rotations.cartesian_to_spherical_numpy(rotated_vecs)
            path= matplotlib.path.Path( np.transpose(np.array([lons,lats])))
            circ_patch = matplotlib.patches.PathPatch(path, transform=ccrs.PlateCarree(), alpha=0.2, **kwargs) 
            circ_artist = axes.add_patch(circ_patch) 
            artists.append(circ_artist)
        artist = axes.scatter(plon,plat, transform=ccrs.PlateCarree(), **kwargs)
        artists.append(artist)
        return artists

class PlateCentroid(Pole):
    """
    Subclass of Pole which represents the centroid
    of a plate. Proxy for plate position (since the
    plate is itself an extended object).
    """
    def __init__(self, longitude, latitude, **kwargs):
        super(PlateCentroid, self).__init__(longitude, latitude, 1.0, **kwargs)


class PaleomagneticPole(Pole):
    """
    Subclass of Pole which represents the centroid
    of a plate. Proxy for plate position (since the
    plate is itself an extended object).
    """
    def __init__(self, longitude, latitude, age, sigma_age=0.0, **kwargs):
        self._age = age
        self._sigma_age = sigma_age
        super(PaleomagneticPole, self).__init__(longitude, latitude, 1.0, **kwargs)

    @property
    def age(self):
        return self._age


class EulerPole(Pole):
    """
    Subclass of Pole which represents an Euler pole.
    The rate is given in deg/Myr
    """
    def __init__(self, longitude, latitude, rate, **kwargs):
        r = rate * rotations.d2r / Julian_year / 1.e6
        super(EulerPole, self).__init__(longitude, latitude, r, **kwargs)
    
    @property
    def rate(self):
        return self.norm * rotations.r2d * Julian_year * 1.e6

    def angle(self, time):
        return self.rate*time

def plot_pole(axes, lon, lat, a95=None, **kwargs):
    artists = []
    if a95 is not None:
	lons = np.linspace(0., 360., 361.)
	lats = np.ones_like(lons)*(90.-a95)
	norms = np.ones_like(lons)
	vecs = rotations.spherical_to_cartesian_numpy(lons,lats,norms)
	rotation_matrix = rotations.construct_euler_rotation_matrix_numpy( 0., (90.-lat)*rotations.d2r, lon*rotations.d2r )
	rotated_vecs = np.dot(rotation_matrix, vecs)
	lons,lats,norms = rotations.cartesian_to_spherical_numpy(rotated_vecs)
	path= matplotlib.path.Path( np.transpose(np.array([lons,lats])))
	circ_patch = matplotlib.patches.PathPatch(path, transform=ccrs.PlateCarree(), alpha=0.2, **kwargs) 
	circ_artist = axes.add_patch(circ_patch) 
	artists.append(circ_artist)
    artist = axes.scatter(lon,lat, transform=ccrs.PlateCarree(), **kwargs)
    artists.append(artist)
    return artists
