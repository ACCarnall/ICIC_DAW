from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import corner
import sys

sys.path.append("../mcmc_sampler")
from mcmc_sampler import mcmc_sampler

# Code to set up the problem and perform the sampling
x = np.array([-1.261, -0.160, 0.334, 0.348, 0.587, 0.860, 1.079])
y = np.array([-0.160, -1.107, 0.472, 0.360, 1.099, 1.321, -0.328])
ex = np.array([-0.587, -0.557, -0.186, -0.222, 0.080, 0.158, 1.540])
dx = np.array([-1.416, -1.221, -1.054, -1.079, -1.012, -0.999, -0.733]) + 1.5

sig = 0.05


def lnlike(param):
    a = param[0]
    b = param[1]
    c = param[2]
    alpha = param[3]/19.8

    model = (a*x + b*y + c + alpha*ex)

    return -0.5*np.sum((model - dx)**2/sig**2)


sampler = mcmc_sampler(lnlike, 4)

n_samples = 200000
prop_width = np.array([0.01, 0.01, 0.01, 0.5])
start_params = np.array([0.16, -0.025, 0.375, 2.5])

sampler.run(n_samples, start_params, prop_width=prop_width)

corner.corner(sampler.samples[int(n_samples/2):, :],
              labels=["$a$", "$b$", "$c$", "$\\alpha$"],
              quantiles=(0.16, 0.5, 0.84), truths=[-999, -999, -999, 1.75])

plt.savefig("corner_mcmc.pdf", bbox_inches="tight")
