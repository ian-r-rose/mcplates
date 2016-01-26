import numpy as np
import matplotlib.pyplot as plt

import pymc3

from mcplates import VonMisesFisher

d2r = np.pi/180.
r2d = 180./np.pi

# Generate a synthetic data set
with pymc3.Model() as synthetic:
    mu_lat = -10.
    mu_lon = 60.
    kappa_hidden = 10

    vmf = VonMisesFisher('vmf', lon_lat=np.array([mu_lon,mu_lat]), kappa=kappa_hidden)
    data = vmf.random(size=100)


with pymc3.Model() as model:
    kappa = pymc3.Exponential('kappa', 1.)
    lon_lat = VonMisesFisher('lon_lat', lon_lat=(0.,0.), kappa=0.00)

    direction = VonMisesFisher('direction', lon_lat=lon_lat, kappa=kappa, observed=data)

    start = pymc3.find_MAP()
    print start
    step = pymc3.Metropolis()

def run(n):
    with model:
        trace = pymc3.sample(n, step, start=start)
        ax = pymc3.traceplot(trace)
        plt.show()

if __name__ == "__main__":
    run(10000)
