import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import mcplates


pole1 = mcplates.PaleomagneticPole( 10., 30., angular_error=30.)
pole2 = mcplates.PaleomagneticPole( 20., 60., angular_error=20.)
pole3 = mcplates.PaleomagneticPole( 30., 80., angular_error=10.)

ax = plt.axes(projection=ccrs.Orthographic(30.,30.))
ax.set_global()
ax.gridlines()

pole1.plot(ax, color='r')
pole2.plot(ax, color='g')
pole3.plot(ax, color='b')

plt.show()
