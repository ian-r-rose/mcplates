import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pymc
import mcplates

dbname = 'siberia.pickle'

data = pd.read_csv("Mongolia_Pole_List.csv")
siberia_data = data[data['Terrane']=='Siberia']

plat = siberia_data['PLat'].values
plon = siberia_data['PLon'].values
a95 = siberia_data['A95'].values
age =  siberia_data['AgeNominal'].values
sigma_age = (siberia_data['AgeUpper'].values-siberia_data['AgeLower'].values)/2.
age_tau = 1./np.power(sigma_age, 2.)
slat = siberia_data['SLat'].values
slon = siberia_data['SLon'].values

for i in range(len(plat)):
    if plon[i] <220.:
        plat[i] = -plat[i]
        plon[i] = np.mod(plon[i]+180., 360.)
n_euler_poles = 2


poles = []
for a, s, lon, lat, error in zip(age, sigma_age, plon, plat, a95):
    pole = mcplates.PaleomagneticPole( lon, lat, angular_error=error, age=a, sigma_age=s) 
    poles.append(pole)

path = mcplates.APWPath( 'siberia', poles, n_euler_poles )

def plot_result( trace ):

    #ax = plt.axes(projection = ccrs.Orthographic(70.,-10.))
    ax = plt.axes(projection = ccrs.Mollweide(0.))
    ax.gridlines()
    ax.set_global()

    direction_samples = path.euler_directions()

    for directions in direction_samples:
        mcplates.plot.plot_distribution( ax, directions[:,0], directions[:,1])


    pathlons, pathlats = path.compute_synthetics(n=100)
    for pathlon,pathlat in zip(pathlons,pathlats):
        ax.plot(pathlon,pathlat, transform=ccrs.PlateCarree(), color='b', alpha=0.05 )

    for lon,lat,a in zip(plon, plat, a95):
        pole = mcplates.PaleomagneticPole( lon, lat, angular_error=a)
        pole.plot(ax)

    plt.show()

if __name__ == "__main__":
    import os 
    if os.path.isfile(path.dbname):
        path.load_mcmc()
    else:
        path.sample_mcmc(10000)
    plot_result(path.db.trace)
