from __future__ import absolute_import
from __future__ import print_function

import copy

import numpy as np
from scipy.constants import Julian_year

import matplotlib.pyplot as plt
import matplotlib.path
import matplotlib.patches

import theano.tensor as tt

import cartopy.crs as ccrs

from . import rotations_theano as rtt
from . import rotations_numpy as rnp

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
        self._pole = rtt.spherical_to_cartesian(longitude, latitude, norm)
        self._angular_error = angular_error

    @property
    def longitude(self):
        return tt.arctan2(self._pole[1], self._pole[0] )*rtt.r2d

    @property
    def latitude(self):
        return 90. - tt.arccos(self._pole[2]/self.norm)*rtt.r2d

    @property
    def colatitude(self):
        return tt.arccos(self._pole[2]/self.norm)*rtt.r2d

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
        lon,lat,norm = rtt.cartesian_to_spherical(pole._pole)
        colat = 90.-lat
        p = rtt.rotate_z(p, -lon[0]*rtt.d2r)
        p = rtt.rotate_y(p, -colat[0]*rtt.d2r)
        p = rtt.rotate_z(p, angle*rtt.d2r)
        p = rtt.rotate_y(p, colat[0]*rtt.d2r)
        self._pole = rtt.rotate_z(p, lon[0]*rtt.d2r)

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
        r = rate * rtt.d2r / Julian_year / 1.e6
        super(EulerPole, self).__init__(longitude, latitude, r, **kwargs)
    
    @property
    def rate(self):
        return self.norm * rtt.r2d * Julian_year * 1.e6

    def angle(self, time):
        return self.rate*time

def plot_pole(axes, lon, lat, a95=None, **kwargs):
    artists = []
    if a95 is not None:
        lons = np.linspace(0., 360., 361.)
        lats = np.ones_like(lons)*(90.-a95)
        norms = np.ones_like(lons)
        vecs = rnp.spherical_to_cartesian(lons,lats,norms)
        rotation_matrix = rnp.construct_euler_rotation_matrix( 0., (90.-lat)*rnp.d2r, lon*rnp.d2r )
        rotated_vecs = np.dot(rotation_matrix, vecs)
        lons,lats,norms = rnp.cartesian_to_spherical(rotated_vecs)
        path= matplotlib.path.Path( np.transpose(np.array([lons,lats])))
        circ_patch = matplotlib.patches.PathPatch(path, transform=ccrs.PlateCarree(), alpha=0.2, **kwargs) 
        circ_artist = axes.add_patch(circ_patch) 
        artists.append(circ_artist)
        artist = axes.scatter(lon,lat, transform=ccrs.PlateCarree(), **kwargs)
        artists.append(artist)
    return artists

def two_sigma_from_kappa( kappa ):
    return 140./np.sqrt(kappa)

def kappa_from_two_sigma( two_sigma ):
    return 140.*140./two_sigma/two_sigma
