import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

import pymc
import mcplates

dbname = 'pickles/invert_euler_pole.pickle'

# Generate a synthetic data set
ages =[0.,10.,20.,30.,40]
sigma_ages = np.array([2., 2., 2., 2., 2.])
age_taus = 1./sigma_ages*sigma_ages
lon_lats = [ [300., -20.], [340.,0.], [0.,30.], [20., 0.], [60.,-20.]]

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

start = mcplates.VonMisesFisher('start', lon_lat=lon_lats[0], kappa=mcplates.kappa_from_two_sigma(10.))
switchpoint = pymc.Uniform('switchpoint', ages[0], ages[-1])

model_vars = [euler_1,rate_1,euler_2,rate_2,start,switchpoint]


for i in range(len(ages)):
    time = pymc.Normal('t'+str(i), mu=ages[i], tau=age_taus[i])
    lon_lat = pymc.Lambda('ll'+str(i), lambda st=start, e1=euler_1, r1=rate_1, e2=euler_2, r2=rate_2, sw=switchpoint, t=time : \
                                              pole_position(st, e1, r1, e2, r2, sw, t),\
                                              dtype=np.float, trace=False, plot=False)
    observed_pole = mcplates.VonMisesFisher('p'+str(i), lon_lat, kappa = mcplates.kappa_from_two_sigma(10.), observed=True, value=lon_lats[i])
    model_vars.append(time)
    model_vars.append(lon_lat)
    model_vars.append(observed_pole)

model = pymc.Model( model_vars )


def sample_mcmc( nsample ):
    mcmc = pymc.MCMC(model, db='pickle', dbname=dbname)
    pymc.MAP(model).fit()
    mcmc.sample(nsample)
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

    interval = max([1,int(len(rates_1)/1000)])

    #ax = plt.axes(projection = ccrs.Orthographic(0.,30.))
    ax = plt.axes(projection = ccrs.Mollweide(0.))
    ax.gridlines()
    ax.set_global()
    mcplates.plot.plot_distribution( ax, euler_1_directions[:,0], euler_1_directions[:,1])
    mcplates.plot.plot_distribution( ax, euler_2_directions[:,0], euler_2_directions[:,1])

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
        pole = mcplates.PaleomagneticPole( p[0], p[1], angular_error=10. )
        pole.plot(ax)


    plt.show()

if __name__ == "__main__":
    import os 
    if os.path.isfile(dbname):
        db = load_mcmc()
    else:
        sample_mcmc(10000)
        db = load_mcmc()
    plot_trace(db.trace)
