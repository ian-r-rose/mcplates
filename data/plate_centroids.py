import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd

d2r = np.pi/180.
r2d = 180./np.pi

def spherical_to_cartesian( longitude, latitude, norm ):
    colatitude = 90.-latitude
    return np.array([ norm * np.sin(colatitude*d2r)*np.cos(longitude*d2r),
                      norm * np.sin(colatitude*d2r)*np.sin(longitude*d2r),
                      norm * np.cos(colatitude*d2r) ] ).T

def cartesian_to_spherical( vecs ):
    v = np.reshape(vecs, (3,-1))
    norm = np.sqrt(v[0,:]*v[0,:] + v[1,:]*v[1,:] + v[2,:]*v[2,:])
    latitude = 90. - np.arccos(v[2,:]/norm)*r2d
    longitude = np.arctan2(v[1,:], v[0,:] )*r2d
    return longitude, latitude, norm

def compute_plate_centroid(lons, lats):

    assert( len(lons) == len(lats) )
    cartesian_pts = spherical_to_cartesian( lons, lats, np.ones_like(lats) )
    cartesian_centroid = np.array([0.,0.,0.])

    for i in range(len(cartesian_pts)-1):
      x1 = cartesian_pts[i] 
      x2 = cartesian_pts[i+1] 
      dx = x2-x1
      r = (x1+x2)/2.
      dx_norm = np.sqrt(np.dot(dx,dx))
      if dx_norm == 0.:
          continue
      dx_hat = dx / dx_norm
      m = np.cross( r, dx_hat )
      cartesian_centroid += m*dx_norm

    lon,lat,norm = cartesian_to_spherical(cartesian_centroid)

    return lon[0], lat[0]

morvel = pd.read_table("NNR-MORVEL56.txt", delim_whitespace=True).set_index('Abbreviation')
#plate_abbreviations = morvel['Abbreviation'].values
plate_abbreviations = ['am', 'ar', 'ca', 'cp', 'in', 'lw', 'nb',
                       'nz', 'ri', 'sc', 'sr', 'sw', 'an', 'au',
                       'co', 'eu', 'jf', 'na', 'mq', 'ps', 'sa',
                       'sm', 'su', 'yz', 'pa']

outfile = open( 'plate_centroids.txt', 'w')
outfile.write('Abbreviation Longitude Latitude\n')

for abbrev in plate_abbreviations:
    data = np.loadtxt("plate_boundaries/"+abbrev, skiprows=1)
    lats = data[:,0]
    lons = data[:,1]
    center_lon, center_lat = compute_plate_centroid( lons, lats )
    elat = morvel['Latitude'][abbrev]
    elon = morvel['Longitude'][abbrev]
    print abbrev, center_lon,center_lat
    outfile.write( "%s\t%f\t%f\n"%(abbrev, center_lon, center_lat))

#    ax = plt.axes(projection=ccrs.Orthographic(center_lon, center_lat))
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.set_global()
    ax.gridlines()
    ax.coastlines(resolution='110m')
    ax.scatter( lons, lats, transform=ccrs.PlateCarree())
    ax.scatter(center_lon, center_lat, transform=ccrs.PlateCarree())
    ax.scatter(elon, elat, transform = ccrs.PlateCarree() , s=30, color='r')
    plt.show()
    plt.clf()

outfile.close()
