import numpy as np
import scipy.stats as st
import scipy.special as sp

import pymc
from pymc.Node import ZeroProbability

d2r = np.pi / 180.
r2d = 180. / np.pi
eps = 1.e-6


def vmf_random(lon_lat, kappa):
    # make the appropriate euler rotation matrix
    alpha = 0.
    beta = np.pi / 2. - lon_lat[1] * d2r
    gamma = lon_lat[0] * d2r
    rot_alpha = np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                          [np.sin(alpha), np.cos(alpha), 0.],
                          [0., 0., 1.]])
    rot_beta = np.array([[np.cos(beta), 0., np.sin(beta)],
                         [0., 1., 0.],
                         [-np.sin(beta), 0., np.cos(beta)]])
    rot_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0.],
                          [np.sin(gamma), np.cos(gamma), 0.],
                          [0., 0., 1.]])
    rotation_matrix = np.dot(rot_gamma, np.dot(rot_beta, rot_alpha))

    def cartesian_sample_generator():
        # Generate samples around the z-axis, then rotate
        # to the appropriate position using euler angles

        # z-coordinate is determined by inversion of the cumulative
        # distribution function for that coordinate.
        zeta = st.uniform.rvs(loc=0., scale=1.)
        if kappa < eps:
            z = 2. * zeta - 1.
        else:
            z = 1. + 1. / kappa * \
                np.log(zeta + (1. - zeta) * np.exp(-2. * kappa))

        # x and y coordinates can be determined by a
        # uniform distribution in longitude.
        phi = st.uniform.rvs(loc=0., scale=2. * np.pi)
        x = np.sqrt(1. - z * z) * np.cos(phi)
        y = np.sqrt(1. - z * z) * np.sin(phi)

        # Rotate the samples to have the correct mean direction
        unrotated_samples = np.array([x, y, z])
        samples = np.transpose(np.dot(rotation_matrix, unrotated_samples))
        return samples

    s = cartesian_sample_generator()
    lon_lat = np.array([np.arctan2(s[1], s[0]), np.pi /
                        2. - np.arccos(s[2] / np.sqrt(np.dot(s, s)))]) * r2d
    return lon_lat


def vmf_logp(x, lon_lat, kappa):

    if x[1] < -90. or x[1] > 90.:
        raise ZeroProbability
        return -np.inf

    if kappa < eps:
        return np.log(1. / 4. / np.pi)

    mu = np.array([np.cos(lon_lat[1] * d2r) * np.cos(lon_lat[0] * d2r),
                   np.cos(lon_lat[1] * d2r) * np.sin(lon_lat[0] * d2r),
                   np.sin(lon_lat[1] * d2r)])
    test_point = np.transpose(np.array([np.cos(x[1] * d2r) * np.cos(x[0] * d2r),
                                        np.cos(x[1] * d2r) *
                                        np.sin(x[0] * d2r),
                                        np.sin(x[1] * d2r)]))

    logp_elem = np.log( -kappa / ( 2. * np.pi * np.expm1(-2. * kappa)) ) + \
        kappa * (np.dot(test_point, mu) - 1.)

    logp = logp_elem.sum()
    return logp

VonMisesFisher = pymc.stochastic_from_dist('von_mises_fisher',
                                           logp=vmf_logp,
                                           random=vmf_random,
                                           dtype=np.float,
                                           mv=True)


def spherical_beta_random(lon_lat, alpha):
    # make the appropriate euler rotation matrix
    beta = np.pi / 2. - lon_lat[1] * d2r
    gamma = lon_lat[0] * d2r
    rot_beta = np.array([[np.cos(beta), 0., np.sin(beta)],
                         [0., 1., 0.],
                         [-np.sin(beta), 0., np.cos(beta)]])
    rot_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0.],
                          [np.sin(gamma), np.cos(gamma), 0.],
                          [0., 0., 1.]])
    rotation_matrix = np.dot(rot_gamma, rot_beta)

    # Generate samples around the z-axis, then rotate
    # to the appropriate position using euler angles

    # z-coordinate is determined by beta distribution
    # with alpha=beta, shifted to go from -1 to 1
    z = 2. * st.beta.rvs(alpha, alpha) - 1.

    # x and y coordinates can be determined by a
    # uniform distribution in longitude.
    phi = st.uniform.rvs(loc=0., scale=2. * np.pi)
    x = np.sqrt(1. - z * z) * np.cos(phi)
    y = np.sqrt(1. - z * z) * np.sin(phi)

    # Rotate the samples to have the correct mean direction
    unrotated_samples = np.array([x, y, z])
    s = np.transpose(np.dot(rotation_matrix, unrotated_samples))

    lon_lat = np.array([np.arctan2(s[1], s[0]), np.pi /
                        2. - np.arccos(s[2] / np.sqrt(np.dot(s, s)))]) * r2d
    return lon_lat


def spherical_beta_logp(x, lon_lat, alpha):

    if x[1] < -90. or x[1] > 90.:
        raise ZeroProbability
        return -np.inf

    if alpha == 1.0:
        return np.log(1. / 4. / np.pi)

    mu = np.array([np.cos(lon_lat[1] * d2r) * np.cos(lon_lat[0] * d2r),
                   np.cos(lon_lat[1] * d2r) * np.sin(lon_lat[0] * d2r),
                   np.sin(lon_lat[1] * d2r)])
    test_point = np.transpose(np.array([np.cos(x[1] * d2r) * np.cos(x[0] * d2r),
                                        np.cos(x[1] * d2r) *
                                        np.sin(x[0] * d2r),
                                        np.sin(x[1] * d2r)]))

    thetas = np.arccos(np.dot(test_point, mu))
    normalization = sp.gamma(alpha + 0.5) / \
        sp.gamma(alpha) / np.sqrt(np.pi) / np.pi / 2.
    logp_elem = np.log(np.sin(thetas)) * (2. * alpha - 2) + \
        np.log(normalization)

    logp = logp_elem.sum()
    return logp

SphericalBeta = pymc.stochastic_from_dist('watson',
                                          logp=spherical_beta_logp,
                                          random=spherical_beta_random,
                                          dtype=np.float,
                                          mv=True)


def watson_girdle_random(lon_lat, kappa):
    # make the appropriate euler rotation matrix
    beta = np.pi / 2. - lon_lat[1] * d2r
    gamma = lon_lat[0] * d2r
    rot_beta = np.array([[np.cos(beta), 0., np.sin(beta)],
                         [0., 1., 0.],
                         [-np.sin(beta), 0., np.cos(beta)]])
    rot_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0.],
                          [np.sin(gamma), np.cos(gamma), 0.],
                          [0., 0., 1.]])
    rotation_matrix = np.dot(rot_gamma, rot_beta)

    # Generate samples around the z-axis, then rotate
    # to the appropriate position using euler angles

    # z-coordinate is determined by beta distribution
    # with alpha=beta, shifted to go from -1 to 1
    if np.abs(kappa) < eps:
        z = st.uniform.rvs(loc=-1., scale=2.)
    else:
        sigma = np.sqrt(-1. / 2. / kappa)
        mu = 0.0
        z = st.truncnorm.rvs(a=(-1. - mu) / sigma,
                             b=(1. - mu) / sigma, loc=0., scale=sigma)

    # x and y coordinates can be determined by a
    # uniform distribution in longitude.
    phi = st.uniform.rvs(loc=0., scale=2. * np.pi)
    x = np.sqrt(1. - z * z) * np.cos(phi)
    y = np.sqrt(1. - z * z) * np.sin(phi)

    # Rotate the samples to have the correct mean direction
    unrotated_samples = np.array([x, y, z])
    s = np.transpose(np.dot(rotation_matrix, unrotated_samples))

    lon_lat = np.array([np.arctan2(s[1], s[0]), np.pi /
                        2. - np.arccos(s[2] / np.sqrt(np.dot(s, s)))]) * r2d
    return lon_lat


def watson_girdle_logp(x, lon_lat, kappa):

    if x[1] < -90. or x[1] > 90.:
        raise ZeroProbability
        return -np.inf

    if np.abs(kappa) < eps:
        return np.log(1. / 4. / np.pi)

    mu = np.array([np.cos(lon_lat[1] * d2r) * np.cos(lon_lat[0] * d2r),
                   np.cos(lon_lat[1] * d2r) * np.sin(lon_lat[0] * d2r),
                   np.sin(lon_lat[1] * d2r)])
    test_point = np.transpose(np.array([np.cos(x[1] * d2r) * np.cos(x[0] * d2r),
                                        np.cos(x[1] * d2r) *
                                        np.sin(x[0] * d2r),
                                        np.sin(x[1] * d2r)]))

    normalization = 1. / sp.hyp1f1(0.5, 1.5, kappa) / 4. / np.pi
    logp_elem = np.log( normalization ) + \
        kappa * (np.dot(test_point, mu)**2.)

    logp = logp_elem.sum()
    return logp


WatsonGirdle = pymc.stochastic_from_dist('watson_girdle',
                                         logp=watson_girdle_logp,
                                         random=watson_girdle_random,
                                         dtype=np.float,
                                         mv=True)
