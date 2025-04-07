import yaml
import torch
from torch.utils.data import DataLoader
from models.unet import Unet
import os
from tqdm import tqdm
import numpy as np
from torch.optim import Adam
from utils.noise_scheduler import LinearNoiseScheduler
from dataloader.mnist_dataloader import MnistDataset
import argparse

def train(args):
    #have a config file and read that
    with open(args.config_path, 'r') as file:
        try:
            config=yaml.safe_load(file)
        except yaml.YAMLError as err:
            print(err)
    print(config)
    
    dataset_config=config['dataset_params']
    diffusion_config=config['diffusion_params']
    model_config=config['model_params']
    train_config=config['train_params']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'], 
                                     beta_start=diffusion_config['beta_start'], 
                                     beta_end=diffusion_config['beta_end'],
                                     device=device)
    
    #create dataset
    mnist = MnistDataset('train', im_path=dataset_config['im_path'])

    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)
    
    #instantiate model
    model = Unet(model_config['im_channels'])
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    #create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    #load checkpoint if found
    checkpoint_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(checkpoint_path):
        try:
            print("Found a checkpoint. Loading the checkpoint")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            print("Successfully loaded checkpoint")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
        
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()
    
    #training loop
    for epoch in range(num_epochs):
        losses=[]
        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            
            #sample random noise and random time step t
            noise = torch.randn_like(im).to(device)
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device) #get random timestep and project as a vector of same dimension as image height
            
            #add noise to im
            noisy_im = scheduler.add_noise(im, noise, t)
            
            #predict the noise to be removed while going backward
            noise_prediction = model(noisy_im, t)
            
            loss = criterion(noise_prediction, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        # Print epoch stats and save checkpoint once per epoch
        print(f'Epoch:{epoch+1} | Loss : {np.mean(losses)}')
        torch.save(model.state_dict(), os.path.join(train_config['task_name'], train_config['ckpt_name']))
    print('Done Training ...')      

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)
            
    