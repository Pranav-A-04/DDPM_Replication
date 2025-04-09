# **DESCRIPTION OF DIFFUSION PROCESS**

The goal is to essentially go from a known distribution to a target distribution using a diffusion process.
The input data (image for example) can be represented as a probability distribution. The goal of the forward process modelled by the following equation:

<p align="center">
    <img src="https://latex.codecogs.com/svg.image?\color{white}$$q(x_t\mid&space;x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t\mathbf{I})$$" />
</p>

is to convert this distribution into a gaussian distribution or normal distribution with mean 0, variance 1.

This can be rewritten as shown below using the reparameterization trick:

<p align="center">
    <img src="https://latex.codecogs.com/svg.image?\color{white}$$x_t=\sqrt{1-\beta_t}x_{t-1}&plus;\sqrt{\beta_t}\epsilon\,\quad\epsilon\sim\mathcal{N}(0,\mathbf{I})$$" />
</p>

The reverse process aims to essentially move from this gaussian distribution back to the original distribution using the reverse diffusion process modelled by the following equation:

<p align="center">
    <img src="https://latex.codecogs.com/svg.image?\color{white}$$p_\theta(x_{t-1}\mid&space;x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))$$" style=""/>
</p>

To go from gaussian noise to the target diffusion, we need to predict the mean and variance of the distribution at the immediately previous timestep to predict the distribution at that timestep. Then we repeate this over and over again until we get to time t=0, at which time we had the orginal input data i.e the target data. Thus, we obtain the distribution of the target data.

# **RESULTS AFTER TRAINING ON MNIST DATASET**

<img src="/results/compressed.gif" alt="Mnist Generated Images" />

The above GIF shows the predictions made by the DDPM model on test data from the mnist dataset over a series of 1000 time steps i.e from t=1000 to t=0.

## Initial Noise (t=1000)

<img src="/results/initial_noise.png" alt="Initial Noise" />

## Final Generated Digits (t=0)

<img src="/results/generated_digits.png" alt="Generated Digits" />

The model successfully transforms pure Gaussian noise into recognizable MNIST digits through the reverse diffusion process.

# **TRAINING AND SAMPLING**
To train the code on the mnist dataset:
```
python train.py --config default.yaml
```

Sampling:
```
python sample.py --config default.yaml
```

To train on any other dataset, first write dataloader, and make appropriate changes in the training script.

# **PREPARING MNIST DATASET**
Download the mnist train and test csvs from:
[Link To Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

Extract the mnist images by running:
```
python utils/extract_mnist.py \
--mnist_train_csv_path <path_to_training_csv> \
--mnist_test_csv_path <path_to_testing_csv> \
--mnist_train_images_path <path_to_store_training_images> \
--mnist_test_images_path <path_to_store_testing_images>
```
If the args are not specified, code expects paths to be same as default paths defined in extract_mnist.py

# **Pre-trained Mnist Ckpt**
You can sample directly using the pretrained model provided in the ckpt/ directory.


# **To-Do / Future Additions**
1. Add Colab demo notebook
2. Train on other datasets like CIFAR 10


# **Citation**
Based on the foundational work: [Ho et al., 2020 - Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)