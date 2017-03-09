import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pymc

from mcplates import VonMisesFisher

# Generate a synthetic data set
lat_hidden = -10.
lon_hidden = 60.
kappa_hidden = 10

vmf = VonMisesFisher('vmf', lon_lat=[lon_hidden,lat_hidden], kappa=kappa_hidden)
data = np.array([vmf.random() for i in range(100)])


model_parameters = []
kappa = pymc.Exponential('kappa', 1.)
lon_lat = VonMisesFisher('lon_lat', lon_lat=(0.,0.), kappa=0.00)
model_parameters.append(kappa)
model_parameters.append(lon_lat)

for sample in data:
    model_parameters.append(VonMisesFisher('direction', lon_lat=lon_lat, kappa=kappa, value=sample, observed=True))

model =pymc.Model(model_parameters)
mcmc = pymc.MCMC(model)
mcmc.sample(10000, 1000, 1)
kappa_trace = mcmc.trace('kappa')[:]
lon_trace = np.mod(mcmc.trace('lon_lat')[:,0],360.)
lat_trace = mcmc.trace('lon_lat')[:,1]

pymc.Matplot.trace(mcmc.trace('lon_lat'))
plt.show()
pymc.Matplot.trace(mcmc.trace('kappa'))
plt.show()

ax = plt.axes(projection = ccrs.Robinson(lon_hidden))
ax.scatter(lon_trace, lat_trace, transform=ccrs.PlateCarree())
ax.gridlines()
ax.set_global()
plt.show()
