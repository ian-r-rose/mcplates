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
ages =[0.,20.,30.]
lon_lats = [ [300., -30.], [360.,0.], [60.,30.] ]
p1 = PaleomagneticPole(longitude = lon_lats[0][0], latitude=lon_lats[0][1], age=ages[0] )
p2 = PaleomagneticPole(longitude = lon_lats[1][0], latitude=lon_lats[1][1], age=ages[1] )
p3 = PaleomagneticPole(longitude = lon_lats[2][0], latitude=lon_lats[2][1], age=ages[2] )
paleopoles = [p1,p2,p3]

paleopoles.pop()
ages.pop()

d2r = np.pi/180.
r2d = 180./np.pi

def generate_pole( euler_pole, age ):
    pole = PaleomagneticPole(lon_lats[0][0], lon_lats[0][1], age=age)
    euler_pole = EulerPole( euler_pole_direction[0], euler_pole_direction[1], euler_pole_rate)
    pole.rotate(euler_pole, euler_pole.rate*age)
    lon = pole.longitude
    lat = pole.latitude
    return tt.as_tensor_variable([lon,lat])

    
with pymc3.Model() as model:
    euler_pole_direction = VonMisesFisher('direction', lon_lat=(0.,0.), kappa=0.00)
    euler_pole_rate = pymc3.Exponential('rate', 1.) 
    euler_pole = EulerPole( euler_pole_direction[0], euler_pole_direction[1], euler_pole_rate)

    for i in range(len(ages)):
        lon_lat = generate_pole(euler_pole, ages[i])
        observed_pole = VonMisesFisher('p'+str(i), lon_lat, kappa = 100., observed=lon_lats[i])
        

    start = pymc3.find_MAP()
    print start
    step = pymc3.Metropolis()

def run(n):
    with model:
        trace = pymc3.sample(n, step, start=start)
        print trace['direction']
        lons_lats = trace['direction']
        ax = pymc3.traceplot(trace)
        plt.show()
        ax = plt.axes(projection = ccrs.Robinson())
        ax.scatter(lons_lats[:,0], lons_lats[:,1], transform=ccrs.PlateCarree())
        ax.gridlines()
        ax.set_global()
        for p in lon_lats:
            plot_pole(ax, p[0], p[1], a95=10.)
        plt.show()

if __name__ == "__main__":
    run(10000)
