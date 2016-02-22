import numpy as np
import numpy.polynomial.legendre as legendre
import matplotlib.pyplot as plt
import pymc
import cartopy.crs as ccrs

import mcplates

d2r = np.pi/180.
r2d = 180./np.pi

mu_lat = 60.
mu_lon =30.
kappa=-2.5


sb = mcplates.WatsonGirdle('sb', lon_lat=(mu_lon,mu_lat), kappa=kappa)
samples = np.array([sb.random() for i in range(200)])
print samples
phi = samples[:,0]
theta = samples[:,1]

n = 90
points, lat_weights = legendre.leggauss(n)
reg_lat = (np.pi/2. - np.arccos( points ))*180./np.pi
reg_lon = np.linspace(0., 2.*np.pi, 2*n+1, endpoint = True )*180./np.pi
mesh_lon, mesh_lat = np.meshgrid(reg_lon, reg_lat)
mesh_weights = np.outer( lat_weights, np.ones_like(reg_lon) )*np.pi/n
mesh_weights[:,-1] = 0.0  # Set duplicated column to zero in weights
mesh_vals = np.empty_like(mesh_weights)
for i, lat in enumerate(reg_lat):
    for j,lon in enumerate(reg_lon):
        mesh_vals[i,j] = np.exp(mcplates.watson_girdle_logp( np.array([lon,lat]), np.array([mu_lon, mu_lat]), kappa ))


tot = np.sum( mesh_vals * mesh_weights ) 
print tot

ax = plt.axes(projection = ccrs.Mollweide())
c = ax.pcolormesh(mesh_lon,mesh_lat, mesh_vals, cmap='copper', transform=ccrs.PlateCarree())
ax.scatter(phi, theta, transform=ccrs.PlateCarree())
ax.gridlines()
ax.set_global()
plt.colorbar(c)
plt.show()

