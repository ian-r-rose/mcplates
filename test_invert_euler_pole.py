import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import theano.tensor as tt
import theano

import pymc
from mcplates import *
import mcplates.rotations_numpy as rnp

# Generate a synthetic data set
#ages =[0.,20.,30.,60.]
#sigma_ages = [2., 2., 2., 2.]
#lon_lats = [ [300., -30.], [360.,0.], [60.,30.], [110., 0.]]
ages =[0.,10.,20.,30.,40]
sigma_ages = np.array([2., 2., 2., 2., 2.])
age_taus = 1./sigma_ages*sigma_ages
lon_lats = [ [300., -20.], [340.,0.], [0.,30.], [20., 0.], [60.,-20.]]

def pole_position( start, euler_1, rate_1, euler_2, rate_2, switchpoint, time ):
    euler_pole_1 = EulerPole( euler_1[0], euler_1[1], rate_1)
    euler_pole_2 = EulerPole( euler_2[0], euler_2[1], rate_2)
    start_pole = PaleomagneticPole(start[0], start[1], age=time)

    if time <= switchpoint:
        start_pole.rotate( euler_pole_1, euler_pole_1.rate*time)
    else:
        start_pole.rotate( euler_pole_1, euler_pole_1.rate*switchpoint)
        start_pole.rotate( euler_pole_2, euler_pole_2.rate*(time-switchpoint))

    lon_lat = np.array([start_pole.longitude, start_pole.latitude])
    return lon_lat


euler_1 = VonMisesFisher('euler_1', lon_lat=(0.,0.), kappa=0.00)
rate_1 = pymc.Exponential('rate_1', 1.) 
euler_2 = VonMisesFisher('euler_2', lon_lat=(0.,0.), kappa=0.00)
rate_2 = pymc.Exponential('rate_2', 1.) 

start = VonMisesFisher('start', lon_lat=lon_lats[0], kappa=kappa_from_two_sigma(10.))
switchpoint = pymc.Uniform('switchpoint', ages[0], ages[-1])

model_vars = [euler_1,rate_1,euler_2,rate_2,start,switchpoint]


for i in range(len(ages)):
    time = pymc.Normal('t'+str(i), mu=ages[i], tau=age_taus[i])
    lon_lat = pymc.Lambda('ll'+str(i), lambda st=start, e1=euler_1, r1=rate_1, e2=euler_2, r2=rate_2, sw=switchpoint, t=time : \
                                              pole_position(st, e1, r1, e2, r2, sw, t),\
                                              dtype=np.float, trace=False, plot=False)
    observed_pole = VonMisesFisher('p'+str(i), lon_lat, kappa = kappa_from_two_sigma(10.), observed=True, value=lon_lats[i])
    model_vars.append(time)
    model_vars.append(lon_lat)
    model_vars.append(observed_pole)

model = pymc.Model( model_vars )
mcmc = pymc.MCMC(model)
pymc.MAP(model).fit()
mcmc.sample(10000, 1000, 1)


euler_1_directions = mcmc.trace('euler_1')[:]
rates_1 = mcmc.trace('rate_1')[:]

euler_2_directions = mcmc.trace('euler_2')[:]
rates_2 = mcmc.trace('rate_2')[:]

start_directions = mcmc.trace('start')[:]
switchpoints = mcmc.trace('switchpoint')[:]

interval = int(len(rates_1)/1000)

ax = plt.axes(projection = ccrs.Orthographic(0.,30.))
#ax = plt.axes(projection = ccrs.Mollweide(0.))
ax.scatter(euler_1_directions[::interval,0], euler_1_directions[::interval,1], transform=ccrs.PlateCarree(), color='r', edgecolors='none', alpha=0.1)
ax.scatter(euler_2_directions[::interval,0], euler_2_directions[::interval,1], transform=ccrs.PlateCarree(), color='g', edgecolors='none', alpha=0.1)
ax.gridlines()
ax.set_global()

interval=1
age_list = np.linspace(ages[0], ages[-1], 100)
pathlons = np.empty_like(age_list)
pathlats = np.empty_like(age_list)
for start, e1, r1, e2, r2, switch \
             in zip(start_directions[::interval], 
                    euler_1_directions[::interval], rates_1[::interval],
                    euler_2_directions[::interval], rates_2[::interval],
                    switchpoints[::interval]):
    for i,a in enumerate(age_list):
        lon_lat = pole_position( start, e1, r1, e2, r2, switch, a)
        pathlons[i] = lon_lat[0]
        pathlats[i] = lon_lat[1]

    ax.plot(pathlons,pathlats,color='b', transform=ccrs.PlateCarree(), alpha=0.05)

for p in lon_lats:
    pole = PaleomagneticPole( p[0], p[1], angular_error=10. )
    pole.plot(ax)


plt.show()
