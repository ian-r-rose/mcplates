import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import mcplates


pole1 = ( 10., 30., 30.)
pole2 = ( 20., 60., 20.)
pole3 = ( 30., 80., 10.)


ax = plt.axes(projection=ccrs.Orthographic(30.,30.))
ax.set_global()
ax.gridlines()
mcplates.plot_pole(ax, pole1[0], pole1[1], a95=pole1[2], color='r')
mcplates.plot_pole(ax, pole2[0], pole2[1], a95=pole2[2], color='g')
mcplates.plot_pole(ax, pole3[0], pole3[1], a95=pole3[2], color='b')

plt.show()
