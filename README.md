# **DESCRIPTION OF DIFFUSION PROCESS** #

The goal is to essentially go from a known distribution to a target distribution using a diffusion process.
The input data (image for example) can be represented as a probability distribution. The goal of the forward process modelled by the following equation:

$$ q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t \, \mathbf{I}) $$

is to convert this distribution into a gaussian distribution or normal distribution with mean 0, variance 1.

This can be rewritten as shown below using the reparameterization trick:
$$ x_t = \sqrt{1 - \beta_t} \, x_{t-1} + \sqrt{\beta_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I}) $$

The reverse process aims to essentially move from this gaussian distribution back to the original distribution using the reverse diffusion process modelled by the following equation:
$$ p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) $$

To go from gaussian noise to the target diffusion, we need to predict the mean and variance of the distribution at the immediately previous timestep to predict the distribution at that timestep. Then we repeate this over and over again until we get to time t=0, at which time we had the orginal input data i.e the target data. Thus, we obtain the distribution of the target data.
