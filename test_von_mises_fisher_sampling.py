import numpy as np
import matplotlib.pyplot as plt
import pymc3
import cartopy.crs as ccrs

import bayesian_plate_tectonics as bpt

d2r = np.pi/180.
r2d = 180./np.pi

mu_colat = 60.
mu_lon =30.
kappa = 50.0


with pymc3.Model():
    vmf = bpt.VonMisesFisher('vmf', lon_colat=(mu_lon,mu_colat), kappa=kappa)
    samples = vmf.random(size=100)
    print samples
    phi = samples[:,0]
    theta = 90.-samples[:,1]

    ax = plt.axes(projection = ccrs.Orthographic(30,30))
    ax.scatter(phi, theta, transform=ccrs.PlateCarree())
    ax.gridlines()
    ax.set_global()
    plt.show()
