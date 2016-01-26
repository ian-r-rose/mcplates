import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
import theano

import pymc3
from mcplates import *
import mcplates.rotations_numpy as rnp

# Generate a synthetic data set
ages =[0.,20.,30.,60.]
lon_lats = [ [300., -30.], [360.,0.], [60.,30.], [110., 0.]]

def generate_pole( start_pole, euler_pole, age ):
    start_pole.rotate(euler_pole, euler_pole.rate*age)
    return [start_pole.longitude,start_pole.latitude]

    
with pymc3.Model() as model:
    euler_pole_direction = VonMisesFisher('euler_pole', lon_lat=(0.,0.), kappa=0.00)
    euler_pole_rate = pymc3.Exponential('rate', 1.) 
    initial_pole_direction = VonMisesFisher('initial_pole', lon_lat=lon_lats[0], kappa=kappa_from_two_sigma(10.))

    euler_pole = EulerPole( euler_pole_direction[0], euler_pole_direction[1], euler_pole_rate)

    for i in range(len(ages)):
        start_pole = PaleomagneticPole(initial_pole_direction[0], initial_pole_direction[1], age=ages[i])
        lon_lat = generate_pole(start_pole, euler_pole, ages[i])
        observed_pole = VonMisesFisher('p'+str(i), lon_lat, kappa = kappa_from_two_sigma(10.), observed=lon_lats[i])
        

    start = pymc3.find_MAP()
    print start
    step = pymc3.Metropolis()

def run(n):
    with model:
        trace = pymc3.sample(n, step, start=start)
        euler_directions = trace['euler_pole']
        start_directions = trace['initial_pole']
        rates = trace['rate']
        elons = euler_directions[:,0]
        elats = euler_directions[:,1]
        slons = start_directions[:,0]
        slats = start_directions[:,1]
#        ax = pymc3.traceplot(trace)
#        plt.show()
        ax = plt.axes(projection = ccrs.Robinson())
        ax.scatter(elons, elats, transform=ccrs.PlateCarree(), edgecolors='none', alpha=0.1)
        ax.gridlines()
        ax.set_global()

        interval = int(len(rates)/1000)
        age_list = np.linspace(ages[0], ages[-1], 100)
        pathlons = np.empty_like(age_list)
        pathlats = np.empty_like(age_list)
        for slon, slat, elon, elat, rate in zip(slons[::interval], slats[::interval], elons[::interval],elats[::interval],rates[::interval]):
            initial_pole = rnp.spherical_to_cartesian( slon, slat, 1.0 )
            euler_pole = rnp.spherical_to_cartesian( elon, elat, rate )
            for i,a in enumerate(age_list):
                final_pole = rnp.rotate(initial_pole, euler_pole, rate*a)
                lon,lat,_ = rnp.cartesian_to_spherical(final_pole)
                pathlons[i] = lon[0]
                pathlats[i] = lat[0]
            ax.plot(pathlons,pathlats,color='r', transform=ccrs.PlateCarree(), alpha=0.1)

        for p in lon_lats:
            plot_pole(ax, p[0], p[1], a95=10.)

        
        plt.show()

if __name__ == "__main__":
    run(10000)
