import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt

import pymc
from mcplates import *
import mcplates.rotations_numpy as rnp

data = pd.read_csv("Kewee_Track_Poles.csv")
data.sort('age', ascending=False, inplace=True)

plat = data['pole_lat'].values
plon = data['pole_lon'].values - 180. # Shift to get around plotting bug
a95 = data['A_95'].values
age =  data['age'].values
#sigma_age =  data['age_error'].values
sigma_age = np.ones_like(age)*2.
age_tau = 1./np.power(sigma_age, 2.)
colors = data['plot_color']

slat = 46.8 # Duluth lat
slon = 360. - 92.1 - 180. # Duluth lon

start_age = max(age)
#start_age = min(age)


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

start = VonMisesFisher('start', lon_lat=(plon[0], plat[0]), kappa=kappa_from_two_sigma(a95[0]))
#start = VonMisesFisher('start', lon_lat=(plon[-1], plat[-1]), kappa=kappa_from_two_sigma(a95[-1]))
switchpoint = pymc.Uniform('switchpoint', age[-1], age[0])

model_vars = [euler_1,rate_1,euler_2,rate_2,start,switchpoint]


for i in range(len(age)):
    pole_age = pymc.Normal('t'+str(i), mu=age[i], tau=age_tau[i])
    lon_lat = pymc.Lambda('ll'+str(i), lambda st=start, e1=euler_1, r1=rate_1, e2=euler_2, r2=rate_2, sw=(start_age-switchpoint), t=(start_age-pole_age) : \
                                              pole_position(st, e1, r1, e2, r2, sw, t),\
                                              dtype=np.float, trace=False, plot=False)
    observed_pole = VonMisesFisher('p'+str(i), lon_lat, kappa = kappa_from_two_sigma(a95[i]), observed=True, value=(plon[i],plat[i]))
    model_vars.append(pole_age)
    model_vars.append(lon_lat)
    model_vars.append(observed_pole)

model = pymc.Model( model_vars )
mcmc = pymc.MCMC(model)
pymc.MAP(model).fit()
#mcmc.sample(100000, 10000, 1)
mcmc.sample(10000, 1000, 1)


euler_1_directions = mcmc.trace('euler_1')[:]
rates_1 = mcmc.trace('rate_1')[:]

euler_2_directions = mcmc.trace('euler_2')[:]
rates_2 = mcmc.trace('rate_2')[:]

start_directions = mcmc.trace('start')[:]
switchpoints = mcmc.trace('switchpoint')[:]

interval = int(len(rates_1)/1000)

ax = plt.axes(projection = ccrs.Orthographic(60.,-10.))
#ax = plt.axes(projection = ccrs.Mollweide(190.-180.))
ax.scatter(euler_1_directions[:,0], euler_1_directions[:,1], transform=ccrs.PlateCarree(), color='r', edgecolors='none', alpha=0.1)
ax.scatter(euler_2_directions[:,0], euler_2_directions[:,1], transform=ccrs.PlateCarree(), color='b', edgecolors='none', alpha=0.1)
ax.gridlines()
ax.set_global()

age_list = np.linspace(age[0], age[-1], 100)
pathlons = np.empty_like(age_list)
pathlats = np.empty_like(age_list)
for start, e1, r1, e2, r2, switch \
             in zip(start_directions[::interval], 
                    euler_1_directions[::interval], rates_1[::interval],
                    euler_2_directions[::interval], rates_2[::interval],
                    switchpoints[::interval]):
    switch_index = np.argmin( np.abs( age_list - switch ) )
    for i,a in enumerate(age_list):
#        lon_lat = pole_position( (slon,slat), e1, r1, e2, r2, (start_age-switch), (start_age-a))
        lon_lat = pole_position( start, e1, r1, e2, r2, (start_age-switch), (start_age-a))
        pathlons[i] = lon_lat[0]
        pathlats[i] = lon_lat[1]

    ax.plot(pathlons[:switch_index],pathlats[:switch_index],color='r', transform=ccrs.PlateCarree(), alpha=0.05)
    ax.plot(pathlons[switch_index:],pathlats[switch_index:],color='b', transform=ccrs.PlateCarree(), alpha=0.05)

for lon,lat,a,c in zip(plon, plat, a95, colors):
    pole = PaleomagneticPole( lon, lat, angular_error=a)
    pole.plot(ax, color=c)


plt.savefig('keweenawan.pdf')
plt.show()



