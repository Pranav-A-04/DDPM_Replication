# **DESCRIPTION OF DIFFUSION PROCESS**

The goal is to essentially go from a known distribution to a target distribution using a diffusion process.
The input data (image for example) can be represented as a probability distribution. The goal of the forward process modelled by the following equation:

$$ q(x*t \mid x*{t-1}) = \mathcal{N}(x*t; \sqrt{1 - \beta_t} x*{t-1}, \beta_t \mathbf{I}) $$

is to convert this distribution into a gaussian distribution or normal distribution with mean 0, variance 1.

This can be rewritten as shown below using the reparameterization trick:

$$ x*t = \sqrt{1 - \beta_t} x*{t-1} + \sqrt{\beta_t} \epsilon \, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I}) $$

The reverse process aims to essentially move from this gaussian distribution back to the original distribution using the reverse diffusion process modelled by the following equation:

$$ p*\theta(x*{t-1} \mid x*t) = \mathcal{N}(x*{t-1}; \mu*\theta(x_t, t), \Sigma*\theta(x_t, t)) $$

To go from gaussian noise to the target diffusion, we need to predict the mean and variance of the distribution at the immediately previous timestep to predict the distribution at that timestep. Then we repeate this over and over again until we get to time t=0, at which time we had the orginal input data i.e the target data. Thus, we obtain the distribution of the target data.

# **RESULTS AFTER TRAINING ON MNIST DATASET**

![Mnist Generated Images](/results/compressed.gif)

The above GIF shows the predictions made by the DDPM model on test data from the mnist dataset over a series of 1000 time steps i.e from t=1000 to t=0.

## Initial Noise (t=1000)

![Initial Noise](/results/initial_noise.png)

## Final Generated Digits (t=0)

![Generated Digits](/results/generated_digits.png)

The model successfully transforms pure Gaussian noise into recognizable MNIST digits through the reverse diffusion process.
