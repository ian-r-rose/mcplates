import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

import pymc
import mcplates

dbname = 'pickles/apw.pickle'

# Generate a synthetic data set
ages =[0.,10.,20.,30.,40]
sigma_ages = np.array([2., 2., 2., 2., 2.])
age_taus = 1./sigma_ages*sigma_ages
lon_lats = [ [300., -20.], [340.,0.], [0.,30.], [20., 0.], [60.,-20.]]

poles = []
for a, s, ll in zip(ages, sigma_ages, lon_lats):
    pole = mcplates.PaleomagneticPole( ll[0], ll[1], angular_error=10., age=a, sigma_age=s) 
    poles.append(pole)

path = mcplates.APWPath( 'apw', poles, 2 )
path.create_model()

def plot_result( trace ):

    ax = plt.axes(projection = ccrs.Orthographic(0.,-30.))
    #ax = plt.axes(projection = ccrs.Mollweide(0.))
    ax.gridlines()
    ax.set_global()

    direction_samples = path.euler_directions()
    for directions in direction_samples:
        mcplates.plot.plot_distribution( ax, directions[:,0], directions[:,1])

    pathlons, pathlats = path.compute_synthetics(n=100)
    for pathlon,pathlat in zip(pathlons,pathlats):
        ax.plot(pathlon,pathlat, transform=ccrs.PlateCarree(), color='b', alpha=0.05 )

    for p in lon_lats:
        pole = mcplates.PaleomagneticPole( p[0], p[1], angular_error=10. )
        pole.plot(ax)


    plt.show()

if __name__ == "__main__":
    import os 
    if os.path.isfile(path.dbname):
        path.load_mcmc()
    else:
        path.sample_mcmc(10000)
    plot_result(path.db.trace)
