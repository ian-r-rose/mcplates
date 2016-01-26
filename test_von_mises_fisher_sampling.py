import numpy as np
import matplotlib.pyplot as plt
import pymc3
import cartopy.crs as ccrs

import mcplates

d2r = np.pi/180.
r2d = 180./np.pi

mu_lat = 30.
mu_lon =30.
kappa = 50.0


with pymc3.Model():
    vmf = mcplates.VonMisesFisher('vmf', lon_lat=(mu_lon,mu_lat), kappa=kappa)
    samples = vmf.random(size=100)
    print samples
    phi = samples[:,0]
    theta = samples[:,1]

    ax = plt.axes(projection = ccrs.Orthographic(30,30))
    ax.scatter(phi, theta, transform=ccrs.PlateCarree())
    ax.gridlines()
    ax.set_global()
    plt.show()
