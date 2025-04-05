import torch
import torch.nn as nn
class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas=torch.linspace(beta_start, beta_end, num_timesteps) #linear schedule going from beta_start to beta_end
        self.alphas=1-self.betas
        self.alphas_cumprod=torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod=torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alpha_cumprod=torch.sqrt(1-self.alphas_cumprod)

    #forward process
    def add_noise(self, originial, noise, t):
        original_shape=originial.shape
        batch_size=original_shape[0]
        sqrt_alpha_cumprod_t=self.sqrt_alpha_cumprod[t].reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alpha_cumprod[t].reshape(batch_size, 1, 1, 1)
        noisy_image=sqrt_alpha_cumprod_t*originial+sqrt_one_minus_alpha_cumprod_t*noise
        return noisy_image
    
    #reverse process
    def sample_prev_timestep(self, xt, t, noise_pred):
        x0=(xt-(self.sqrt_one_minus_alpha_cumprod[t]*noise_pred))/self.sqrt_alpha_cumprod[t]
        x0=torch.clamp(x0, -1., 1.)

        mean=xt-((self.betas[t]*noise_pred)/self.sqrt_one_minus_alpha_cumprod[t])
        mean=mean/torch.sqrt(self.alphas[t])

        if(t==0):
            return mean, x0
        else:
            variance=(self.betas[t]*(1-self.alphas_cumprod[t-1]))/(1-self.alphas_cumprod[t])
            sigma=variance**0.5
            z=torch.randn(xt.shape)
            return mean+sigma*z, x0

    

