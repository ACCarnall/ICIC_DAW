from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner
import sys
import lochnest_monster as nesty

sys.path.append("../preliminary")
from exercises import *


def lnlike(param):
    omega_m = param[0]
    h = param[1]

    model_dist_mod = calc_dist_mod(data["z"], omega_m, h)

    mtaked = np.matrix(data["mu"] - model_dist_mod)

    chisq = np.tensordot(np.tensordot(mtaked, cov_inv, axes=1), mtaked)

    return -0.5*chisq


def prior_trans(cube):
    return cube


fname = "jla_mub.txt"
data = pd.read_table(fname, delimiter=" ", skiprows=1,
                     names=open(fname).readline()[1:].split())

cov = np.matrix(np.loadtxt("jla_mub_covmatrix").reshape(31, 31))
cov_inv = np.linalg.inv(cov)

sampler = nesty.ellipsoid_sampler(lnlike, prior_trans, 2, n_live=2500)
sampler.run()

corner.corner(sampler.results["samples_eq"],
              labels=["$\\Omega_\\mathrm{M}$", "$h$"])

plt.savefig("corner_nesty.pdf", bbox_inches="tight")
