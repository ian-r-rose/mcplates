import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

d2r = np.pi/180.
r2d = 180./np.pi

def spherical_to_cartesian( longitude, latitude, norm ):
    colatitude = 90.-latitude
    return np.array([ norm * np.sin(colatitude*d2r)*np.cos(longitude*d2r),
                      norm * np.sin(colatitude*d2r)*np.sin(longitude*d2r),
                      norm * np.cos(colatitude*d2r) ] ).T

morvel = pd.read_table("NNR-MORVEL56.txt", delim_whitespace=True).set_index('Abbreviation')
centroids = pd.read_table("plate_centroids.txt", delim_whitespace=True).set_index('Abbreviation')

#Plate abbreviations for the MORVEL 25 largest plates
plate_abbreviations = ['am', 'ar', 'ca', 'cp', 'in', 'lw', 'nb',
                       'nz', 'ri', 'sc', 'sr', 'sw', 'an', 'au',
                       'co', 'eu', 'jf', 'na', 'mq', 'ps', 'sa',
                       'sm', 'su', 'yz', 'pa']

angles = []

for abbrev in plate_abbreviations:

    plat = centroids['Latitude'][abbrev]
    plon = centroids['Longitude'][abbrev]

    elat = morvel['Latitude'][abbrev]
    elon = morvel['Longitude'][abbrev]
    
    pvec = spherical_to_cartesian( plon, plat, 1.)
    evec = spherical_to_cartesian( elon, elat, 1.)

    angles.append( np.arccos(np.dot(pvec, evec))*r2d )

plt.hist(angles, bins=[0,15,45,75,105,135,165,180])
plt.show()
