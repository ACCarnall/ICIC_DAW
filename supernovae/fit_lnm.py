from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lochnest_monster as lnm
import corner

nu_coefs = (1, 0.154, 0.4304, 0.19097, 0.066941)


def calc_nu(a, omega_m):
    s = np.cbrt((1-omega_m)/omega_m)

    return 2*np.sqrt(s**3 + 1)*(nu_coefs[0]/a**4
                                + nu_coefs[1]*s/a**3
                                + nu_coefs[2]*s**2/a**2
                                + nu_coefs[3]*s**3/a
                                + nu_coefs[4]*s**4)**(-1/8)


def calc_ldist(z, omega_m):
    a = 1/(1+z)
    nu_0 = calc_nu(1, omega_m)
    nu_a = calc_nu(a, omega_m)

    return 3000*(1+z)*(nu_0 - nu_a)


def calc_dist_mod(z, omega_m, h):
    ldist = calc_ldist(z, omega_m)

    return 25 - 5*np.log10(h) + 5*np.log10(ldist)


def lnlike(param):
    omega_m = param[0]
    h = param[1]

    model_dist_mod = calc_dist_mod(data["z"], omega_m, h)

    chisq = np.sum(((data["mu"] - model_dist_mod)/data["mu_err"])**2)

    return -0.5*chisq


def prior_trans(cube):
    return cube


fname = "jla_mub.txt"
data = pd.read_table(fname, delimiter=" ", skiprows=1,
                      names=open(fname).readline()[1:].split())

data["mu_err"] = 0.1


ndim = 2

sampler = lnm.combined_sampler(lnlike, prior_trans, ndim)
sampler.run()

corner.corner(sampler.results["samples_eq"], labels=["$\\Omega_\\mathrm{M}$", "$h$"])
plt.savefig("corner.pdf", bbox_inches="tight")
