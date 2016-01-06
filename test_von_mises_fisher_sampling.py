import numpy as np
import matplotlib.pyplot as plt
import pymc3
import cartopy.crs as ccrs

import bayesian_plate_tectonics as bpt

d2r = np.pi/180.
r2d = 180./np.pi

mu_colat = 110.
mu_lon = 20.
mu = np.array([np.sin(mu_colat*d2r)*np.cos(mu_lon*d2r),\
	       np.sin(mu_colat*d2r)*np.sin(mu_lon*d2r),\
	       np.cos(mu_colat*d2r)])
kappa = 50.0


with pymc3.Model():
    vmf = bpt.VonMisesFisher('vmf', mu=mu, kappa=kappa)
    samples = vmf.random(size=1000)
    x = samples[:,0]
    y = samples[:,1]
    z = samples[:,2]
    r = np.sqrt(x*x+y*y+z*z)
    theta = 90.-np.arccos(z/r)*r2d
    phi = np.arctan2(y,x)*r2d

    ax = plt.axes(projection = ccrs.Orthographic(30.,30.))
    ax.scatter(phi, theta, transform=ccrs.PlateCarree())
    ax.gridlines()
    ax.set_global()
    plt.show()
