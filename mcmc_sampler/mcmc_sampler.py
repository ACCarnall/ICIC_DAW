from __future__ import print_function, division, absolute_import

import numpy as np


class mcmc_sampler(object):
    """ Implement a basic MCMC routine.

    Parameters
    ----------

    lnprob : function
        A function which takes an array of parameter values and returns
        the natural log of the posterior probability at that point.

    n_dim : int
        The number of free parameters you wish to fit.
    """

    def __init__(self, lnprob, n_dim):
        self.n_dim = n_dim
        self.user_lnprob = lnprob

    def lnprob(self, input_param):
        """ Wrapper on the user's lnprob function. """

        param = np.copy(input_param)

        return self.user_lnprob(param)

    def run(self, n_samples, p0, prop_width=0.1):
        """ Run the sampler.

        Parameters
        ----------

        n_samples : function
            Number of samples to draw from the posterior.

        p0 : numpy.ndarray
            Starting parameter vector.

        prop_width : int or numpy.ndarray
            Standard deviation of the Gaussian proposal function, can be
            a float or array of floats with length n_dim.
        """

        self.samples = np.zeros((n_samples, self.n_dim))
        self.lnprobs = np.zeros(n_samples)

        self.samples[0, :] = p0
        self.lnprobs[0] = self.lnprob(p0)

        self.accepted = 0

        for i in range(1, n_samples):
            prop_offset = prop_width*np.random.randn(self.n_dim)
            new_sample = self.samples[i-1, :] + prop_offset
            p = np.random.rand()
            new_lnprob = self.lnprob(new_sample)

            if np.log(p) <= new_lnprob - self.lnprobs[i-1]:
                self.samples[i, :] = new_sample
                self.lnprobs[i] = new_lnprob
                self.accepted += 1

            else:
                self.samples[i, :] = self.samples[i-1, :]
                self.lnprobs[i] = self.lnprobs[i-1]

        print("Acceptance fraction: ", self.accepted/n_samples)
