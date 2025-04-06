import yaml
import torch
import os
from models.unet import Unet
from utils.noise_scheduler import LinearNoiseScheduler
from tqdm import tqdm
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#sampling
def sample(model, scheduler, train_config, model_config, diffusion_config):
    xt=torch.randn((train_config['num_samples'],
                    model_config['im_channels'],
                    model_config['im_size'],
                    model_config['im_size']
                    )).to(device)
    
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        #get the noise pred
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        #sample prev timestep
        xt_1, x0_pred = scheduler.sample_prev_timestep(xt, torch.as_tensor(i).to(device), noise_pred)
        
        #save x0 at each step to see progression of prediction
        ims = torch.clamp(x0_pred, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage(grid)
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        
        img.save(os.path.join(train_config['task_name'], 'samples', f'x0_{i}.png'))
        img.close()
        
        #set new xt to xt-1(i.e xt_1)
        xt=xt_1
        
        
def infer(args):
    with open(args.config_path, 'r') as file:
        try:
            config=yaml.safe_load(file)
        except yaml.YAMLError as err:
            print(err)
    print(config)
    
    diffusion_config=config['diffusion_params']
    model_config=config['model_params']
    train_config=config['train_params']
    
    #load model checkpoint
    
    model = Unet(model_config['im_channels']).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['ckpt_name']), map_location=device))
    
    model.eval()
    
    #noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'], 
                                     beta_start=diffusion_config['beta_start'], 
                                     beta_end=diffusion_config['beta_end'])
    
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
    