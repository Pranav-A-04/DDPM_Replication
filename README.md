# **DESCRIPTION OF DIFFUSION PROCESS** #

The goal is to essentially go from a known distribution to a target distribution using a diffusion process.
The input data (image for example) can be represented as a probability distribution. The goal of the forward process modelled by the following equation:

is to convert this distribution into a gaussian distribution or normal distribution with mean 0, variance 1.

The reverse process aims to essentially move from this gaussian distribution back to the original distribution using the reverse diffusion process modelled by the following equation:

To go from gaussian noise to the target diffusion, we need to predict the mean and variance of the distribution at the immediately previous timestep to predict the distribution at that timestep. Then we repeate this over and over again until we get to time t=0, at which time we had the orginal input data i.e the target data. Thus, we obtain the distribution of the target data.

# **EXPLANATION OF HOW WE OBTAIN THE TRAINING OBJECTIVE:** #

Let us consider the predicted distribution to be represented by P(xt), where x is the input data at some timestep t.

Thus, P(x0) represents the reconstructed distribution at t=0 i.e it represents the target distribution.
Our training objective would thus be to maximize the log likelihood of P(x0), essentially means that we want to maximize the probability that all of x0's parameters are predicted correctly(i.e the mean & variance).
