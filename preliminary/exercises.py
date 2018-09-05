from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

if __name__ is "__main__":

    """ Exercise 1 """

    z = np.arange(0.01, 2, 0.01)
    omega_m_list = (0.2, 0.3, 0.4, 0.5)

    plt.figure()
    for omega_m in omega_m_list:
        dist_mod = calc_dist_mod(z, omega_m, 0.7)
        plt.plot(z, dist_mod, label="$" + str(omega_m) + "$")

    plt.xlim(0, 2)
    plt.ylim(32.5, 47.5)
    plt.xlabel("$\\mathrm{Redshift}$")
    plt.ylabel("$\\mathrm{Distance\\ Modulus}$")
    plt.legend(title="$\\Omega_m$", frameon=False)

    """ Exercise 2 """

    fname = "jla_mub.txt"
    table = pd.read_table(fname, delimiter=" ", skiprows=1,
                          names=open(fname).readline()[1:].split())

    plt.scatter(table["z"], table["mu"], color="purple", s=10, linewidth=0.5,
                edgecolor="black", zorder=10)

    plt.savefig("1_observational_data.pdf", bbox_inches="tight")
    plt.close()

    """ Exercise 3 """


    def simulate_sne(n=20, z_range=(0, 2), omega_m=0.3, h=0.7):
        z_vals = (z_range[1] - z_range[0])*np.random.rand(n) + z_range[0]
        dist_mod_vals = calc_dist_mod(z_vals, omega_m, h)

        return pd.DataFrame(np.c_[z_vals, dist_mod_vals], columns=("z", "mu"))


    sim_table = simulate_sne()

    """ Exercise 4 """

    sim_table.loc[:, "mu"] += 0.1*np.random.randn(sim_table.shape[0])

    """ Exercise 5 """

    h_list = (0.6, 0.7, 0.8)

    plt.figure()
    for h in h_list:
        dist_mod = calc_dist_mod(z, 0.3, h)
        plt.plot(z, dist_mod, label="$" + str(h) + "$")

    plt.xlim(0, 2)
    plt.ylim(32.5, 47.5)
    plt.xlabel("$\\mathrm{Redshift}$")
    plt.ylabel("$\\mathrm{Distance\\ Modulus}$")
    plt.legend(title="$h$", frameon=False)

    plt.errorbar(sim_table["z"], sim_table["mu"], yerr=0.1, lw=0.5, linestyle=" ",
                 capsize=2, capthick=0.5, color="black", zorder=9)

    plt.scatter(sim_table["z"], sim_table["mu"], color="purple", s=10,
                linewidth=0.5, edgecolor="black", zorder=10)

    plt.savefig("2_simulated_data.pdf", bbox_inches="tight")
    plt.close()
