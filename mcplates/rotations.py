import numpy as np

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
