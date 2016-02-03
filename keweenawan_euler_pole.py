import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pymc
import mcplates

dbname = 'keweenawan.pickle'

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


def pole_position( start, euler_1, rate_1, euler_2, rate_2, switchpoint, time ):
    euler_pole_1 = mcplates.EulerPole( euler_1[0], euler_1[1], rate_1)
    euler_pole_2 = mcplates.EulerPole( euler_2[0], euler_2[1], rate_2)
    start_pole = mcplates.PaleomagneticPole(start[0], start[1], age=time)

    if time <= switchpoint:
        start_pole.rotate( euler_pole_1, euler_pole_1.rate*time)
    else:
        start_pole.rotate( euler_pole_1, euler_pole_1.rate*switchpoint)
        start_pole.rotate( euler_pole_2, euler_pole_2.rate*(time-switchpoint))

    lon_lat = np.array([start_pole.longitude, start_pole.latitude])
    return lon_lat


euler_1 = mcplates.VonMisesFisher('euler_1', lon_lat=(0.,0.), kappa=0.00)
rate_1 = pymc.Exponential('rate_1', 1.) 
euler_2 = mcplates.VonMisesFisher('euler_2', lon_lat=(0.,0.), kappa=0.00)
rate_2 = pymc.Exponential('rate_2', 1.) 

start = mcplates.VonMisesFisher('start', lon_lat=(plon[0], plat[0]), kappa=mcplates.kappa_from_two_sigma(a95[0]))
switchpoint = pymc.Uniform('switchpoint', age[-1], age[0])

model_vars = [euler_1,rate_1,euler_2,rate_2,start,switchpoint]


for i in range(len(age)):
    pole_age = pymc.Normal('t'+str(i), mu=age[i], tau=age_tau[i])
    lon_lat = pymc.Lambda('ll'+str(i), lambda st=start, e1=euler_1, r1=rate_1, e2=euler_2, r2=rate_2, sw=(start_age-switchpoint), t=(start_age-pole_age) : \
                                              pole_position(st, e1, r1, e2, r2, sw, t),\
                                              dtype=np.float, trace=False, plot=False)
    observed_pole = mcplates.VonMisesFisher('p'+str(i), lon_lat, kappa = mcplates.kappa_from_two_sigma(a95[i]), observed=True, value=(plon[i],plat[i]))
    model_vars.append(pole_age)
    model_vars.append(lon_lat)
    model_vars.append(observed_pole)

model = pymc.Model( model_vars )

def sample_mcmc( nsample ):
    mcmc = pymc.MCMC(model, db='pickle', dbname=dbname)
    pymc.MAP(model).fit()
    mcmc.sample(nsample, int(nsample/5), 1)
    mcmc.db.close()
    return mcmc.db

def load_mcmc():
    db = pymc.database.pickle.load(dbname)
    return db

def plot_trace( trace ):
    euler_1_directions = trace('euler_1')[:]
    rates_1 = trace('rate_1')[:]

    euler_2_directions = trace('euler_2')[:]
    rates_2 = trace('rate_2')[:]

    start_directions = trace('start')[:]
    switchpoints = trace('switchpoint')[:]

    interval = int(len(rates_1)/300)

    ax = plt.axes(projection = ccrs.Orthographic(60.,-10.))
#    ax = plt.axes(projection = ccrs.Mollweide(80.))
    ax.gridlines()
    ax.set_global()
    mcplates.plot.plot_distribution( ax, euler_1_directions[:,0], euler_1_directions[:,1], cmap=mcplates.plot.cmap_red, resolution=30)
    mcplates.plot.plot_distribution( ax, euler_2_directions[:,0], euler_2_directions[:,1] , cmap=mcplates.plot.cmap_blue, resolution=30)

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
        pole = mcplates.PaleomagneticPole( lon, lat, angular_error=a)
        pole.plot(ax, color=c)

    plt.savefig('keweenawan.pdf')
    plt.show()

    speed_1_samples = np.empty_like(rates_1)
    speed_2_samples = np.empty_like(rates_1)
    index = 0

    for start, e1, r1, e2, r2, switch \
                 in zip(start_directions, 
                        euler_1_directions, rates_1,
                        euler_2_directions, rates_2,
                        switchpoints):

        duluth = mcplates.PlateCentroid( slon, slat )
        euler_pole_1 = mcplates.EulerPole( e1[0], e1[1], r1 )
        euler_pole_2 = mcplates.EulerPole( e2[0], e2[1], r2 )

        speed_1 = euler_pole_1.speed_at_point( duluth )
        duluth.rotate( euler_pole_1, euler_pole_1.rate * (start_age - switch))
        speed_2 = euler_pole_2.speed_at_point( duluth )

        speed_1_samples[index] = speed_1
        speed_2_samples[index] = speed_2
        index += 1

    plt.subplot(121)
    plt.hist( speed_1_samples, bins=30, normed=True )
    plt.ylabel('Probability density')
    plt.xlabel('Speed (cm/yr)')
    plt.subplot(122)
    plt.hist( speed_2_samples, bins=30, normed=True )
    plt.ylabel('Probability density')
    plt.xlabel('Speed (cm/yr)')

    plt.savefig('keweenawan_speeds.pdf')
    plt.show()

if __name__ == "__main__":
    import os 
    if os.path.isfile(dbname):
        db = load_mcmc()
    else:
        sample_mcmc(100000)
        db = load_mcmc()
    plot_trace(db.trace)
