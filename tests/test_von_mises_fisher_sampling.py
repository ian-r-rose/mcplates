import numpy as np
import matplotlib.pyplot as plt
import pymc
import cartopy.crs as ccrs

import mcplates

d2r = np.pi/180.
r2d = 180./np.pi

mu_lat = 30.
mu_lon =30.
kappa = 50.0


vmf = mcplates.VonMisesFisher('vmf', lon_lat=(mu_lon,mu_lat), kappa=kappa)
samples = np.array([vmf.random() for i in range(100)])
print samples
phi = samples[:,0]
theta = samples[:,1]

ax = plt.axes(projection = ccrs.Orthographic(30,30))
ax.scatter(phi, theta, transform=ccrs.PlateCarree())
ax.gridlines()
ax.set_global()
plt.show()
