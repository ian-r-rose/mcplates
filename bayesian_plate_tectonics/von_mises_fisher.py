import numpy as np
import scipy.stats as st

from pymc3.distributions import Continuous
from pymc3.distributions.distribution import draw_values, generate_samples

import theano.tensor

d2r = np.pi/180.
r2d = 180./np.pi
eps = 1.e-6

def construct_euler_rotation_matrix(alpha, beta, gamma):
    """
    Make a 3x3 matrix which represents a rigid body rotation,
    with alpha being the first rotation about the z axis,
    beta being the second rotation about the y axis, and
    gamma being the third rotation about the z axis.
 
    All angles are assumed to be in radians
    """
    rot_alpha = np.array( [ [np.cos(alpha), -np.sin(alpha), 0.],
                            [np.sin(alpha), np.cos(alpha), 0.],
                            [0., 0., 1.] ] )
    rot_beta = np.array( [ [np.cos(beta), 0., np.sin(beta)],
                           [0., 1., 0.],
                           [-np.sin(beta), 0., np.cos(beta)] ] )
    rot_gamma = np.array( [ [np.cos(gamma), -np.sin(gamma), 0.],
                            [np.sin(gamma), np.cos(gamma), 0.],
                            [0., 0., 1.] ] )
    rot = np.dot( rot_gamma, np.dot( rot_beta, rot_alpha ) )
    return rot
    

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

    def __init__(self, mu, kappa, *args, **kwargs):
        super(VonMisesFisher, self).__init__(shape=3, *args, **kwargs)
#        print mu
#        assert(mu.shape == (3,))
        assert(theano.tensor.ge(kappa,0.))
        assert(theano.tensor.le(abs(theano.tensor.sqrt(mu[0]*mu[0]+mu[1]*mu[1]+mu[2]*mu[2]) - 1.), 1.e-8))
        self.mu = mu
        self.median = self.mode = self.mean = self.mu
        self.kappa = kappa
#        self.normalization = kappa / 4. / np.pi / np.sinh(kappa)

    def _pdf(point, mu, kappa):
        """
        Probability density function for Von Mises-Fisher
        distribution.

        This uses a numerically more stable implementation
        for tight concentrations.
        """
        if kappa < eps:
            return 1./4./np.pi
        else:
            return -kappa / ( 2. * np.pi * np.expm1(-2.*kappa) ) * \
                   np.exp( kappa * np.dot(point, mu) )
        #return self.normalization * \
        #       np.exp( self.kappa * np.dot(point, mu) )

    def random(self, point=None, size=None):
        mu, kappa = draw_values([self.mu, self.kappa], point=point)

        # make the appropriate euler rotation matrix
        alpha = 0. # rotationally symmetric distribution, so this does not matter
        beta = np.arccos(mu[2]) #unit vector, norm is one
        gamma = np.arctan2(mu[1], mu[0])
        rotation_matrix = construct_euler_rotation_matrix(alpha, beta, gamma)

        def sample_generator(size=None):
            # Generate samples around the z-axis, then rotate
            # to the appropriate position using euler angles

            # z-coordinate is determined by inversion of the cumulative
            # distribution function for that coordinate.
            zeta = st.uniform.rvs(loc=0., scale=1., size=size)
            #z = 1./kappa * np.log(np.exp(-kappa) + 2.*zeta*np.sinh(kappa))
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
            
        samples = sample_generator(size) 
        return samples

    def logp(self, point):
        #return theano.tensor.log(self.normalization) + self.kappa * theano.tensor.dot(point, self.mu)
        kappa = self.kappa
        mu = self.mu

        if theano.tensor.le(kappa, eps):
            return theano.tensor.log( 1./4./np.pi )
        else:
            return theano.tensor.log( -kappa / ( 2.*np.pi * theano.tensor.expm1(-2.*kappa)) ) + \
                   kappa * (theano.tensor.dot(point,mu)-1.)

