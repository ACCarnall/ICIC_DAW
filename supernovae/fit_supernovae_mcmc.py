from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner
import sys

sys.path.append("../mcmc_sampler")
from mcmc_sampler import mcmc_sampler

sys.path.append("../preliminary")
from exercises import *


def lnlike(param):
    omega_m = param[0]
    h = param[1]

    model_dist_mod = calc_dist_mod(data["z"], omega_m, h)

    mtaked = np.matrix(data["mu"] - model_dist_mod)

    chisq = np.tensordot(np.tensordot(mtaked, cov_inv, axes=1), mtaked)

    return -0.5*chisq


fname = "jla_mub.txt"
data = pd.read_table(fname, delimiter=" ", skiprows=1,
                     names=open(fname).readline()[1:].split())

cov = np.matrix(np.loadtxt("jla_mub_covmatrix").reshape(31, 31))
cov_inv = np.linalg.inv(cov)

sampler = mcmc_sampler(lnlike, 2)

n_samples = 50000
prop_width = 0.01
start_params = np.array([0.3, 0.7])

sampler.run(n_samples, start_params, prop_width=prop_width)

corner.corner(sampler.samples[int(n_samples/2):, :],
              labels=["$\\Omega_\\mathrm{M}$", "$h$"])

plt.savefig("corner_mcmc.pdf", bbox_inches="tight")
