import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import theano.tensor as tt
import theano

import pymc3
from mcplates import *
import mcplates.rotations_numpy as rnp

# Generate a synthetic data set
#ages =[0.,20.,30.,60.]
#sigma_ages = [2., 2., 2., 2.]
#lon_lats = [ [300., -30.], [360.,0.], [60.,30.], [110., 0.]]
ages =[0.,10.,20.,30.,40]
sigma_ages = [2., 2., 2., 2., 2.]
lon_lats = [ [300., -20.], [340.,0.], [0.,30.], [20., 0.], [60.,-20.]]

@theano.compile.ops.as_op(itypes=[tt.dvector, tt.dvector, tt.dscalar, tt.dvector, tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector])
def generate_pole( start, euler_1, rate_1, euler_2, rate_2, switchpoint, time ):
    euler_pole_1 = EulerPole( euler_1[0], euler_1[1], rate_1)
    euler_pole_2 = EulerPole( euler_2[0], euler_2[1], rate_2)
    start_pole = PaleomagneticPole(start[0], start[1], age=time)

    if time <= switchpoint:
        start_pole.rotate( euler_pole_1, euler_pole_1.rate*time)
    else:
        start_pole.rotate( euler_pole_1, euler_pole_1.rate*switchpoint)
        start_pole.rotate( euler_pole_2, euler_pole_2.rate*(time-switchpoint))
    return np.array([start_pole.longitude, start_pole.latitude])
    
generate_pole.grad = lambda *x: x[0]

    
with pymc3.Model() as model:
    euler_1 = VonMisesFisher('euler_1', lon_lat=(0.,0.), kappa=0.00)
    rate_1 = pymc3.Exponential('rate_1', 1.) 
    euler_2 = VonMisesFisher('euler_2', lon_lat=(0.,0.), kappa=0.00)
    rate_2 = pymc3.Exponential('rate_2', 1.) 

    start = VonMisesFisher('start', lon_lat=lon_lats[0], kappa=kappa_from_two_sigma(10.))
    switchpoint = pymc3.Uniform('switchpoint', ages[0], ages[-1])


    for i in range(len(ages)):
        time = pymc3.Normal('t'+str(i), mu=ages[i], sd=sigma_ages[i])
        lon_lat = generate_pole(start, euler_1, rate_1, euler_2, rate_2, switchpoint, time)
        observed_pole = VonMisesFisher('p'+str(i), lon_lat, kappa = kappa_from_two_sigma(10.), observed=lon_lats[i])
        

    start = pymc3.find_MAP(fmin=scipy.optimize.fmin_powell)
    print start
    step = pymc3.Metropolis()

def run(n):
    with model:
        trace = pymc3.sample(n, step, start=start)

        euler_1_directions = trace['euler_1']
        e1lons = euler_1_directions[:,0]
        e1lats = euler_1_directions[:,1]
        rates_1 = trace['rate_1']

        euler_2_directions = trace['euler_2']
        e2lons = euler_2_directions[:,0]
        e2lats = euler_2_directions[:,1]
        rates_2 = trace['rate_2']

        start_directions = trace['start']
        slons = start_directions[:,0]
        slats = start_directions[:,1]

        switchpoints = trace['switchpoint']

        ax = pymc3.traceplot(trace)
        plt.show()

        ax = plt.axes(projection = ccrs.Orthographic(0.,30.))
        ax.scatter(e1lons, e1lats, transform=ccrs.PlateCarree(), color='r', edgecolors='none', alpha=0.1)
        ax.scatter(e2lons, e2lats, transform=ccrs.PlateCarree(), color='g', edgecolors='none', alpha=0.1)
        ax.gridlines()
        ax.set_global()

        interval = int(len(rates_1)/1000)
        age_list = np.linspace(ages[0], ages[-1], 100)
        pathlons = np.empty_like(age_list)
        pathlats = np.empty_like(age_list)
        print e1lons
        print e2lons
        for slon, slat, elon1, elat1, rate1, elon2, elat2, rate2, switch \
                     in zip(slons[::interval], slats[::interval], 
                            e1lons[::interval],e1lats[::interval], rates_1[::interval],
                            e2lons[::interval],e2lats[::interval], rates_2[::interval],
                            switchpoints[::interval]):
            initial_pole = rnp.spherical_to_cartesian( slon, slat, 1.0 )
            euler_pole_1 = rnp.spherical_to_cartesian( elon1, elat1, rate1 )
            euler_pole_2 = rnp.spherical_to_cartesian( elon2, elat2, rate2 )
            for i,a in enumerate(age_list):
                if a < switch:
                    final_pole = rnp.rotate(initial_pole, euler_pole_1, rate1*a)
                else:
                    mid_pole = rnp.rotate(initial_pole, euler_pole_1, rate1*switch)
                    final_pole = rnp.rotate(mid_pole, euler_pole_2, rate2*(a-switch))

                lon,lat,_ = rnp.cartesian_to_spherical(final_pole)
                pathlons[i] = lon[0]
                pathlats[i] = lat[0]

            ax.plot(pathlons,pathlats,color='b', transform=ccrs.PlateCarree(), alpha=0.1)

        for p in lon_lats:
            plot_pole(ax, p[0], p[1], a95=10.)

        
        plt.show()

if __name__ == "__main__":
    run(10000)
