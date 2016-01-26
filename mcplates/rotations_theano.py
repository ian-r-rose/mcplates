import numpy as np
import theano.tensor as tt

d2r = np.pi/180.
r2d = 180./np.pi

def spherical_to_cartesian( longitude, latitude, norm ):
#    assert(tt.all(tt.ge(longitude, 0.)) and tt.all(tt.le(longitude,360.)))
    assert(tt.all(tt.ge(latitude, -90.)) and tt.all(tt.le(latitude,90.)))
    assert(tt.all(tt.ge(norm, 0.)))
    colatitude = 90.-latitude
    return [ norm * tt.sin(colatitude*d2r)*tt.cos(longitude*d2r),
             norm * tt.sin(colatitude*d2r)*tt.sin(longitude*d2r),
             norm * tt.cos(colatitude*d2r) ]

def cartesian_to_spherical( vecs ):
    v = tt.reshape(vecs, (3,-1))
    norm = tt.sqrt(v[0,:]*v[0,:] + v[1,:]*v[1,:] + v[2,:]*v[2,:])
    latitude = 90. - tt.arccos(v[2,:]/norm)*r2d
    longitude = tt.arctan2(v[1,:], v[0,:] )*r2d
    return longitude, latitude, norm

def rotate_x(vector, theta):
    flat_rot = tt.as_tensor_variable([1., 0., 0.,
                                      0., tt.cos(theta), -tt.sin(theta),
                                      0., tt.sin(theta), tt.cos(theta)])
    rot = flat_rot.reshape((3,3))
    return tt.dot(rot, vector.T)

def rotate_y(vector, theta):
    flat_rot = tt.as_tensor_variable([tt.cos(theta), 0., tt.sin(theta),
                                      0., 1., 0.,
                                      -tt.sin(theta), 0., tt.cos(theta)])
    rot = flat_rot.reshape((3,3))
    return tt.dot(rot, vector.T)

def rotate_z(vector, theta):
    flat_rot = tt.as_tensor_variable([tt.cos(theta), -tt.sin(theta), 0.,
                                      tt.sin(theta), tt.cos(theta), 0.,
                                      0., 0., 1.])
    rot = flat_rot.reshape((3,3))
    return tt.dot(rot, vector.T)

def construct_euler_rotation_matrix(alpha, beta, gamma):
    """
    Make a 3x3 matrix which represents a rigid body rotation,
    with alpha being the first rotation about the z axis,
    beta being the second rotation about the y axis, and
    gamma being the third rotation about the z axis.
 
    All angles are assumed to be in radians
    """
    flat_rot_alpha = tt.as_tensor_variable([ tt.cos(alpha), -tt.sin(alpha), 0.,
                                             tt.sin(alpha), tt.cos(alpha), 0.,
                                             0., 0., 1.])
    flat_rot_beta = tt.as_tensor_variable([ tt.cos(beta), 0., tt.sin(beta),
                                            0., 1., 0.,
                                            -tt.sin(beta), 0., tt.cos(beta)])
    flat_rot_gamma = tt.as_tensor_variable([ tt.cos(gamma), -tt.sin(gamma), 0.,
                                             tt.sin(gamma), tt.cos(gamma), 0.,
                                             0., 0., 1.])
    rot_alpha = flat_rot_alpha.reshape((3,3))
    rot_beta = flat_rot_beta.reshape((3,3))
    rot_gamma = flat_rot_gamma.reshape((3,3))

    rot = tt.dot( rot_gamma, tt.dot( rot_beta, rot_alpha ) )
    return rot

def rotate(pole, rotation_pole, angle):
    # The idea is to rotate the pole so that the Euler pole is
    # at the pole of the coordinate system, then perform the
    # requested rotation, then restore things to the original
    # orientation 
    lon,lat,norm = cartesian_to_spherical(rotation_pole)
    colat = 90.-lat
    p = rotate_z(pole, -lon[0]*d2r)
    p = rotate_y(p, -colat[0]*d2r)
    p = rotate_z(p, angle*d2r)
    p = rotate_y(p, colat[0]*d2r)
    return rotate_z(p, lon[0]*d2r)

