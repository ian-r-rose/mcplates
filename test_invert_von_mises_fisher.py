import numpy as np
import matplotlib.pyplot as plt

import pymc3

from mcplates import VonMisesFisher

d2r = np.pi/180.
r2d = 180./np.pi

# Generate a synthetic data set
with pymc3.Model() as synthetic:
    mu_colat = 100.
    mu_lon = 60.
    kappa_hidden = 10

    vmf = VonMisesFisher('vmf', lon_colat=np.array([mu_lon,mu_colat]), kappa=kappa_hidden)
    data = vmf.random(size=100)


with pymc3.Model() as model:
    kappa = pymc3.Exponential('kappa', 1.)
    lon_colat = VonMisesFisher('lon_colat', lon_colat=(0.,0.), kappa=0.01)

    direction = VonMisesFisher('direction', lon_colat=lon_colat, kappa=kappa, observed=data)

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
