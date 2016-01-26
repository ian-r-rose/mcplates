import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
import theano

import pymc3
from mcplates import *
import mcplates.rotations as rotations

d2r = np.pi/180.
r2d = 180./np.pi

# Generate a synthetic data set
ages = np.array([10.,20.,30.])
lon_colats = [ [0., 10.], [0.,20.], [0.,30.] ]
p1 = PaleomagneticPole(longitude = lon_colats[0][0], latitude=lon_colats[0][1], age=ages[0] )
p2 = PaleomagneticPole(longitude = lon_colats[1][0], latitude=lon_colats[1][1], age=ages[1] )
p3 = PaleomagneticPole(longitude = lon_colats[2][0], latitude=lon_colats[2][1], age=ages[2] )
paleopoles = [p1,p2,p3]


d2r = np.pi/180.
r2d = 180./np.pi

def generate_pole( euler_pole_direction, euler_pole_rate, age ):
    #initial_pole = tt.as_tensor_variable( [1.,0.,0.] )
    #euler_pole = rotations.spherical_to_cartesian(euler_pole_direction[0],euler_pole_direction[1], euler_pole_rate)
    #final_pole = rotations.rotate(initial_pole, euler_pole, euler_pole_rate*age) 
    pole = PaleomagneticPole( 0., 0., age=age)
    euler_pole = EulerPole( euler_pole_direction[0], euler_pole_direction[1], euler_pole_rate)
    pole.rotate(euler_pole, age)
    lon = pole.longitude
    colat = pole.colatitude
    #lon,lat,norm = rotations.cartesian_to_spherical(final_pole)
    return [lon,colat]

    
with pymc3.Model() as model:
    euler_pole_direction = VonMisesFisher('direction', lon_colat=(0.,0.), kappa=0.00)
    euler_pole_rate = pymc3.Exponential('rate', 1.) 
    euler_pole = EulerPole( euler_pole_direction[0], 90.-euler_pole_direction[1], euler_pole_rate)

    for i in range(len(ages)):
        lon_colat = generate_pole(euler_pole_direction, euler_pole_rate, ages[i])
        observed_pole = VonMisesFisher('p'+str(i), lon_colat, kappa = 10., observed=lon_colats[i])
        

    start = pymc3.find_MAP()
    print start
    step = pymc3.Metropolis()

def run(n):
    with model:
        trace = pymc3.sample(n, step, start=start)
        print trace['direction']
        lons_colats = trace['direction']
        ax = pymc3.traceplot(trace)
        plt.show()
        ax = plt.axes(projection = ccrs.Robinson())
        ax.scatter(lons_colats[:,0], 90.-lons_colats[:,1], transform=ccrs.PlateCarree())
        ax.gridlines()
        ax.set_global()
        plt.show()

if __name__ == "__main__":
    run(10000)
