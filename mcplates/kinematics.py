import copy
import numpy as np
from scipy.constants import Julian_year
import cartopy.crs as ccrs

d2r = np.pi/180.
r2d = 180./np.pi

def spherical_to_cartesian( longitude, latitude, norm ):
    assert(longitude >= 0. and longitude <= 360.)
    assert(latitude >= -90. and latitude <= 90.)
    assert(norm >= 0.)
    colatitude = 90.-latitude
    return np.array([ norm * np.sin(colatitude*d2r)*np.cos(longitude*d2r),
                      norm * np.sin(colatitude*d2r)*np.sin(longitude*d2r),
                      norm * np.cos(colatitude*d2r) ] )

def cartesian_to_spherical( vec ):
    norm = np.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
    latitude = 90. - np.arccos(vec[2]/norm)*r2d
    longitude = np.arctan2(vec[1], vec[0] )*r2d
    return longitude, latitude, norm

def rotate_x(vector, theta):
    rot = np.array( [ [1., 0., 0.],
                      [0., np.cos(theta), -np.sin(theta)],
                      [0., np.sin(theta), np.cos(theta)] ] )
    return np.dot(rot, vector.T)

def rotate_y(vector, theta):
    rot = np.array( [ [np.cos(theta), 0., np.sin(theta)],
                      [0., 1., 0.],
                      [-np.sin(theta), 0., np.cos(theta)] ] )
    return np.dot(rot, vector.T)

def rotate_z(vector, theta):
    rot = np.array( [ [np.cos(theta), -np.sin(theta), 0.],
                      [np.sin(theta), np.cos(theta), 0.],
                      [0., 0., 1.] ] )
    return np.dot(rot, vector.T)

class Pole(object):
    """
    Class representing a pole on the globe:
    essentially a 3-vector with some additional
    properties and operations.
    """
    def __init__(self, longitude, latitude, norm):
        """
        Initialize the pole with lon, lat, and norm.
        """
        self._pole = spherical_to_cartesian(longitude, latitude, norm)
        self._phi = d2r * longitude
        self._theta = d2r * (90.-latitude)

    @property
    def longitude(self):
        return np.arctan2(self._pole[1], self._pole[0] )*r2d

    @property
    def latitude(self):
        return 90. - np.arccos(self._pole[2]/self.norm)*r2d

    @property
    def colatitude(self):
        return np.arccos(self._pole[2]/self.norm)*r2d

    @property
    def norm(self):
        return np.sqrt(self._pole[0]*self._pole[0] + self._pole[1]*self._pole[1] + self._pole[2]*self._pole[2])

    def copy(self):
        return copy.deepcopy(self)

    def rotate(self, pole, angle):
        # The idea is to rotate the pole so that the Euler pole is
        # at the pole of the coordinate system, then perform the
        # requested rotation, then restore things to the original
        # orientation 
        p = self._pole
        p = rotate_z(p, -pole.longitude*d2r)
        p = rotate_y(p, -pole.colatitude*d2r)
        p = rotate_z(p, angle*d2r)
        p = rotate_y(p, pole.colatitude*d2r)
        self._pole = rotate_z(p, pole.longitude*d2r)

    def plot(self, axes, **kwargs):
        artist = axes.scatter(self.longitude,self.latitude, transform=ccrs.PlateCarree(), **kwargs)
        return artist

class PlateCentroid(Pole):
    """
    Subclass of Pole which represents the centroid
    of a plate. Proxy for plate position (since the
    plate is itself an extended object).
    """
    def __init__(self, longitude, latitude):
        super(PlateCentroid, self).__init__(longitude, latitude, 1.0)


class PaleomagneticPole(Pole):
    """
    Subclass of Pole which represents the centroid
    of a plate. Proxy for plate position (since the
    plate is itself an extended object).
    """
    def __init__(self, longitude, latitude, age, sigma_position=0.0, sigma_age=0.0):
        self.age = age
        self.sigma_position = sigma_position
        self.sigma_age = sigma_age
        super(PaleomagneticPole, self).__init__(longitude, latitude, 1.0)


class EulerPole(Pole):
    """
    Subclass of Pole which represents an Euler pole.
    The rate is given in deg/Myr
    """
    def __init__(self, longitude, latitude, rate):
        r = rate * d2r / Julian_year / 1.e6
        super(EulerPole, self).__init__(longitude, latitude, r)
    
    @property
    def rate(self):
        return self.norm * r2d * Julian_year * 1.e6

    def angle(self, time):
        return self.rate*time
