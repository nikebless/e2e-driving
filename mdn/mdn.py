"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
LOG2PI = math.log(2 * math.pi)


# adapted from:
# - https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py
# - https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/#implementation-tricks-to-ensure-stability
class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Linear(in_features, num_gaussians) 
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians) # TODO: try initializaing the bias to left/straight/right turns

    def forward(self, x):
        pi = self.pi(x)
        sigma = F.elu(self.sigma(x)) + 1 + 1e-15
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(x)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target, log=False):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    if log:
        ret = (
            -torch.log(sigma)
            -0.5 * LOG2PI
            -0.5 * ((target - mu) / sigma) ** 2
        )
    else: 
        ret = (ONEOVERSQRT2PI / sigma) * torch.exp(
            -0.5 * ((target - mu) / sigma) ** 2
        )
    return ret.squeeze()


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target.
    """
    log_component_prob = gaussian_probability(sigma, mu, target, log=True)
    log_mix_prob = torch.log(
        F.gumbel_softmax(pi, tau=1, dim=-1) + 1e-15 # sharper distribution
    )
    log_sum_prob = torch.logsumexp(log_component_prob + log_mix_prob, dim=1)
    return -torch.mean(log_sum_prob)


def sample(pi, sigma, mu, return_variances=False):
    """Draw samples from a MoG. 
    Don't sample, just pick the mean of the most likely one, following the MDN paper (1994).
    """
    # Choose which gaussian we'll sample from
    best_gaussian_idx = pi.argmax(dim=1).view(pi.size(0), 1, 1)

    variance_samples = sigma.gather(1, best_gaussian_idx).detach().squeeze() # B[xO]
    mean_samples = mu.detach().gather(1, best_gaussian_idx).squeeze() # B[xO]

    if return_variances:
        return mean_samples, variance_samples
    return mean_samples