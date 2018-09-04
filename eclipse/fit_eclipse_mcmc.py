from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import corner

class mcmc_sampler(object):
    """ Implement a basic MCMC routine.

    Parameters
    ----------

    lnlike : function
        A function which takes an array of parameter values and returns
        the natural log of the likelihood at that point.

    n_dim : int
        The number of free parameters you wish to fit.
    """

    def __init__(self, lnlike, n_dim):
        self.n_dim = n_dim
        self.user_lnlike = lnlike

    def lnlike(self, input_param):
        """ Wrapper on the user's lnlike function. """

        param = np.copy(input_param)

        return self.user_lnlike(param)

    def run(self, n_samples, p0, prop_width=0.1):
        self.samples = np.zeros((n_samples, self.n_dim))
        self.likelihoods = np.zeros(n_samples)
        self.samples[0,:] = p0
        self.accepted = 0


        for i in range(1, n_samples):
            new_sample = self.proposal(self.samples[i - 1, :], width=prop_width)
            p = np.random.rand()
            new_like = self.lnlike(new_sample)

            if np.log(p) <= new_like - self.likelihoods[i - 1]:
                self.samples[i, :] = new_sample
                self.accepted += 1

            else:
                self.samples[i, :] = self.samples[i - 1, :]

        print("Acceptance fraction: ", self.accepted/n_samples)

    def proposal(self, p0, width=0.1):
        return p0 + width*np.random.randn(self.n_dim)


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

n_samples = 5000000

sampler.run(n_samples, [0.16, -0.025, 0.375, 2.5], prop_width=0.025)

corner.corner(sampler.samples,
              labels=["a", "b", "c", "alpha"])

plt.savefig("eclipse_corner_plot.pdf", bbox_inches="tight")
