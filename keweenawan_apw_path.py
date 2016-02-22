import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pymc
import mcplates

dbname = 'keweenawan_apw.pickle'

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


poles = []
for a, s, lon, lat, error in zip(age, sigma_age, plon, plat, a95):
    pole = mcplates.PaleomagneticPole( lon, lat, angular_error=error, age=a, sigma_age=s) 
    poles.append(pole)

path = mcplates.APWPath( 'keweenawan_apw', poles, 2 )
path.create_model( site_lon_lat=(slon, slat), watson_concentration=-1.0)

def plot_result( trace ):

    ax = plt.axes(projection = ccrs.Orthographic(70.,-10.))
    #ax = plt.axes(projection = ccrs.Mollweide(0.))
    ax.gridlines()
    ax.set_global()

    direction_samples = path.euler_directions()

    for directions in direction_samples:
        mcplates.plot.plot_distribution( ax, directions[:,0], directions[:,1])


    pathlons, pathlats = path.compute_synthetics(n=100)
    for pathlon,pathlat in zip(pathlons,pathlats):
        ax.plot(pathlon,pathlat, transform=ccrs.PlateCarree(), color='b', alpha=0.05 )

    for lon,lat,a,c in zip(plon, plat, a95, colors):
        pole = mcplates.PaleomagneticPole( lon, lat, angular_error=a)
        pole.plot(ax, color=c)

    ax.scatter( slon,slat, transform=ccrs.PlateCarree(), marker="*", s=100)
    plt.show()

if __name__ == "__main__":
    import os 
    if os.path.isfile(path.dbname):
        path.load_mcmc()
    else:
        path.sample_mcmc(100000)
    plot_result(path.db.trace)
