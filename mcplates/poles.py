from __future__ import absolute_import
from __future__ import print_function

import copy

import numpy as np
from scipy.constants import Julian_year

import matplotlib.pyplot as plt
import matplotlib.path
import matplotlib.patches

import cartopy.crs as ccrs

from . import rotations as rot


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
        self._pole = rot.spherical_to_cartesian(longitude, latitude, norm)
        self._pole = np.asarray(self._pole)
        self._angular_error = angular_error

    @property
    def longitude(self):
        return np.arctan2(self._pole[1], self._pole[0]) * rot.r2d

    @property
    def latitude(self):
        return 90. - np.arccos(self._pole[2] / self.norm) * rot.r2d

    @property
    def colatitude(self):
        return np.arccos(self._pole[2] / self.norm) * rot.r2d

    @property
    def norm(self):
        return np.sqrt(self._pole[0] * self._pole[0] + self._pole[1] * self._pole[1] + self._pole[2] * self._pole[2])

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
        p = pole._pole
        lon, lat, norm = rot.cartesian_to_spherical(p)
        colat = 90. - lat
        m1 = rot.construct_euler_rotation_matrix(
            -lon[0] * rot.d2r, -colat[0] * rot.d2r, angle * rot.d2r)
        m2 = rot.construct_euler_rotation_matrix(
            0., colat[0] * rot.d2r, lon[0] * rot.d2r)
        self._pole = np.dot(m2, np.dot(m1, self._pole))

    def add(self, pole):
        self._pole = self._pole + pole._pole

    def plot(self, axes, south_pole=False, **kwargs):
        artists = []
        if self._angular_error is not None:
            lons = np.linspace(0., 360., 361.)
            lats = np.ones_like(lons) * (90. - self._angular_error)
            norms = np.ones_like(lons)
            vecs = rot.spherical_to_cartesian(lons, lats, norms)
            rotation_matrix = rot.construct_euler_rotation_matrix(
                0., (self.colatitude) * rot.d2r, self.longitude * rot.d2r)
            rotated_vecs = np.dot(rotation_matrix, vecs)
            lons, lats, norms = rot.cartesian_to_spherical(rotated_vecs)
            if south_pole is True:
                lons = lons-180.
                lats = -lats
            path = matplotlib.path.Path(np.transpose(np.array([lons, lats])))
            circ_patch = matplotlib.patches.PathPatch(
                path, transform=ccrs.PlateCarree(), alpha=0.5, **kwargs)
            circ_artist = axes.add_patch(circ_patch)
            artists.append(circ_artist)
        if south_pole is False:
            artist = axes.scatter(self.longitude, self.latitude,
                                  transform=ccrs.PlateCarree(), **kwargs)
        else:
            artist = axes.scatter(self.longitude-180., -self.latitude,
                                  transform=ccrs.PlateCarree(), **kwargs)
        artists.append(artist)
        return artists


class PlateCentroid(Pole):
    """
    Subclass of Pole which represents the centroid
    of a plate. Proxy for plate position (since the
    plate is itself an extended object).
    """

    def __init__(self, longitude, latitude, **kwargs):
        super(PlateCentroid, self).__init__(
            longitude, latitude, 6371.e3, **kwargs)


class PaleomagneticPole(Pole):
    """
    Subclass of Pole which represents the centroid
    of a plate. Proxy for plate position (since the
    plate is itself an extended object).
    """

    def __init__(self, longitude, latitude, age=0., sigma_age=0.0, **kwargs):

        if np.iterable(sigma_age) == 1:
            assert len(sigma_age) == 2  # upper and lower bounds
            self._age_type = 'uniform'
        else:
            self._age_type = 'gaussian'

        self._age = age
        self._sigma_age = sigma_age

        super(PaleomagneticPole, self).__init__(
            longitude, latitude, 1.0, **kwargs)

    @property
    def age_type(self):
        return self._age_type

    @property
    def age(self):
        return self._age

    @property
    def sigma_age(self):
        return self._sigma_age


class EulerPole(Pole):
    """
    Subclass of Pole which represents an Euler pole.
    The rate is given in deg/Myr
    """

    def __init__(self, longitude, latitude, rate, **kwargs):
        r = rate * rot.d2r / Julian_year / 1.e6
        super(EulerPole, self).__init__(longitude, latitude, r, **kwargs)

    @property
    def rate(self):
        return self.norm * rot.r2d * Julian_year * 1.e6

    def angle(self, time):
        return self.rate * time

    def speed_at_point(self, pole):
        """
        Given a pole, calculate the speed that the pole
        rotates around the Euler pole. This assumes that
        the test pole has a radius equal to the radius of Earth,
        6371.e3 meters. It returns the speed in cm/yr.
        """
        # Give the point the radius of the earth
        point = pole._pole
        point = point / np.sqrt(np.dot(point, point)) * 6371.e3

        # calculate the speed
        vel = np.cross(self._pole, point)
        speed = np.sqrt(np.dot(vel, vel))

        return speed * Julian_year * 100.


def two_sigma_from_kappa(kappa):
    return 140. / np.sqrt(kappa)


def kappa_from_two_sigma(two_sigma):
    return 140. * 140. / two_sigma / two_sigma
