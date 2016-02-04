import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import pandas as pd
import ssrfpy
import cartopy.crs as ccrs

d2r = np.pi/180.
r2d = 180./np.pi

def cartesian_to_spherical( vecs ):
    v = np.reshape(vecs, (3,-1))
    norm = np.sqrt(v[0,:]*v[0,:] + v[1,:]*v[1,:] + v[2,:]*v[2,:])
    latitude = 90. - np.arccos(v[2,:]/norm)*r2d
    longitude = np.arctan2(v[1,:], v[0,:] )*r2d
    return longitude[0], latitude[0], norm[0]

def spherical_to_cartesian( longitude, latitude, norm ):
    colatitude = 90.-latitude
    return np.array([ norm * np.sin(colatitude*d2r)*np.cos(longitude*d2r),
                      norm * np.sin(colatitude*d2r)*np.sin(longitude*d2r),
                      norm * np.cos(colatitude*d2r) ] ).T

plate_id_to_code = { 0 : 'an',
                     1 : 'au',
                     2 : 'nb',
                     3 : 'pa',
                     4 : 'eu',
                     5 : 'na',
                     6 : 'nz',
                     7 : 'co',
                     8 : 'ca',
                     9 : 'ar',
                    10 : 'ps',
                    11 : 'sa',
                    12 : 'in',
                    13 : 'jf' }
                     

morvel = pd.read_table("NNR-MORVEL56.txt", delim_whitespace=True).set_index('Abbreviation')
plate_data = np.loadtxt("WhichPlate.dat")
lons = plate_data[:,0]
lats = plate_data[:,1]
vals = plate_data[:,2]

nlons = 256.
nlats = 128.
dlon = 360./nlons
dlat = 180./nlats

lons = lons.reshape(nlats,nlons)
lats = lats.reshape(nlats,nlons)
lats = 90. - lats
vals = vals.reshape(nlats,nlons)


n_samples = 100000
val_samples = np.zeros(n_samples)
lon_samples = np.zeros(n_samples)
lat_samples = np.zeros(n_samples)
z_samples = np.zeros(n_samples)

i=0
while i < n_samples:
    x = np.array([1.,1.,1.])
    while np.dot(x,x) > 1.:
        x = 2.*random.random(3)-1.
    lon,lat,norm = cartesian_to_spherical( x )
    try:
        lon_index = np.floor(lon/dlon)
        lat_index = np.floor((90.-lat)/dlat)
        plate_id = vals[lat_index, lon_index]
        plate_code = plate_id_to_code[plate_id]

        elat = morvel['Latitude'][plate_code]
        elon = morvel['Longitude'][plate_code]
        evec = spherical_to_cartesian( elon, elat, 1.)
        lon_samples[i] = lon
        lat_samples[i] = lat
        z_samples[i]  = np.sin(lat*d2r)
        val_samples[i] =  np.arccos(np.dot(x, evec))*r2d
        #val_samples[i] =  np.dot(x, evec)
        i += 1
    except KeyError:
        continue

#ax = plt.axes( projection=ccrs.Mollweide(0.) )
#c = ax.pcolormesh(lons,lats,vals, transform=ccrs.PlateCarree())
#ax.scatter(lon_samples,lat_samples, c=val_samples, transform=ccrs.PlateCarree())
#plt.colorbar(c)
#plt.show()
#uniform_x = np.linspace(-1., 1., 100)
#uniform_y = np.ones_like(uniform_x)*0.5
uniform_x = np.linspace(0., 180., 100)
uniform_y = np.sin(uniform_x*d2r)*0.5*d2r
plt.hist( val_samples, bins=10, normed=True )
plt.plot( uniform_x, uniform_y, 'r', lw=3)
plt.show()
