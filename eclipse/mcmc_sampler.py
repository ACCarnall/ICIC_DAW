from __future__ import print_function, division, absolute_import

import numpy as np


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
        self.samples[0, :] = p0
        self.accepted = 0

        for i in range(1, n_samples):
            new_sample = self.proposal(self.samples[i-1, :], width=prop_width)
            p = np.random.rand()
            new_like = self.lnlike(new_sample)

            if np.log(p) <= new_like - self.likelihoods[i-1]:
                self.samples[i, :] = new_sample
                self.accepted += 1

            else:
                self.samples[i, :] = self.samples[i-1, :]

        print("Acceptance fraction: ", self.accepted/n_samples)

    def proposal(self, p0, width=0.1):
        return p0 + width*np.random.randn(self.n_dim)
