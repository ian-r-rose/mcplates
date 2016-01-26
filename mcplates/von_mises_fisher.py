import numpy as np
import scipy.stats as st

from pymc3.distributions import Continuous
from pymc3.distributions.distribution import draw_values, generate_samples
from pymc3.distributions.dist_math import bound

import theano.tensor as tt

d2r = np.pi/180.
r2d = 180./np.pi
eps = 1.e-6

class VonMisesFisher(Continuous):
    """
    Von Mises-Fisher distribution

    Parameters
    ----------

    mu : cartesian unit vector representing
        the mean direction
    kappa : floating point number representing the
        spread of the distribution.
    """

    def __init__(self, lon_lat, kappa, *args, **kwargs):
        super(VonMisesFisher, self).__init__(shape=2, *args, **kwargs)

        assert(tt.ge(kappa,0.))

        lon = lon_lat[0]*d2r
        lat = lon_lat[1]*d2r
        self.lon_lat = lon_lat
        self.mu = [ tt.cos(lat) * tt.cos(lon),
                    tt.cos(lat) * tt.sin(lon),
                    tt.sin(lat) ]
        self.kappa = kappa
        self.median = self.mode = self.mean = lon_lat

    def random(self, point=None, size=None):
        lon_lat, kappa = draw_values([self.lon_lat, self.kappa], point=point)
        # make the appropriate euler rotation matrix
        alpha = 0.
        beta = np.pi/2. - lon_lat[1]*d2r
        gamma = lon_lat[0]*d2r
        rot_alpha = np.array( [ [np.cos(alpha), -np.sin(alpha), 0.],
                                [np.sin(alpha), np.cos(alpha), 0.],
                                [0., 0., 1.] ] )
        rot_beta = np.array( [ [np.cos(beta), 0., np.sin(beta)],
                               [0., 1., 0.],
                               [-np.sin(beta), 0., np.cos(beta)] ] )
        rot_gamma = np.array( [ [np.cos(gamma), -np.sin(gamma), 0.],
                                [np.sin(gamma), np.cos(gamma), 0.],
                                [0., 0., 1.] ] )
        rotation_matrix = np.dot( rot_gamma, np.dot( rot_beta, rot_alpha ) )

        def cartesian_sample_generator(size=None):
            # Generate samples around the z-axis, then rotate
            # to the appropriate position using euler angles

            # z-coordinate is determined by inversion of the cumulative
            # distribution function for that coordinate.
            zeta = st.uniform.rvs(loc=0., scale=1., size=size)
            if kappa < eps:
                z = 2.*zeta-1.
            else:
                z = 1. + 1./kappa * np.log(zeta + (1.-zeta)*np.exp(-2.*kappa) )

            # x and y coordinates can be determined by a 
            # uniform distribution in longitude.
            phi = st.uniform.rvs(loc=0., scale=2.*np.pi, size=size)
            x = np.sqrt(1.-z*z)*np.cos(phi)
            y = np.sqrt(1.-z*z)*np.sin(phi)

            # Rotate the samples to have the correct mean direction
            unrotated_samples = np.vstack([x,y,z])
            samples = np.transpose(np.dot(rotation_matrix, unrotated_samples))
            return samples
            
        cartesian_samples = cartesian_sample_generator(size) 
        count = size if size is not None else 1
        lat_samples = np.fromiter( (np.pi/2. - np.arccos( s[2]/np.sqrt(np.dot(s,s)) ) for s in cartesian_samples), dtype=np.float64, count=size)
        lon_samples = np.fromiter( (np.arctan2( s[1], s[0] ) for s in cartesian_samples), dtype=np.float64, count=size)
        return np.transpose(np.vstack((lon_samples, lat_samples)))*r2d

    def logp(self, lon_lat):
        kappa = self.kappa
        mu = self.mu
        lon_lat_r = tt.reshape( lon_lat*d2r, (-1, 2) )
        point = [ tt.cos(lon_lat_r[:,1]) * tt.cos(lon_lat_r[:,0]),
                  tt.cos(lon_lat_r[:,1]) * tt.sin(lon_lat_r[:,0]),
                  tt.sin(lon_lat_r[:,1]) ]
        point = tt.as_tensor_variable(point).T

        return bound( tt.switch( tt.ge(kappa, eps), \
                                             # Kappa greater than zero
                                             tt.log( -kappa / ( 2.*np.pi * tt.expm1(-2.*kappa)) ) + \
                                             kappa * (tt.dot(point,mu)-1.),
                                             # Kappa equals zero
                                             tt.log(1./4./np.pi)),
                      tt.all( lon_lat_r[:,1] >= -np.pi/2. ),
                      tt.all( lon_lat_r[:,1] <= np.pi/2. ) )

