from torch.utils.data import Dataset
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import os
import glob
import PIL
from PIL import Image

class MnistDataset(Dataset):
    def __init__(self, split, im_path, im_ext='png'):
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)
        
    def load_images(self, im_path):
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, f"*.{self.im_ext}")):
                ims.append(fname)
                labels.append(int(d_name))
        print(f"found {len(ims)} images for split {self.split}")
        return ims, labels #use labels when conditioning
    
    def __len__(self):
        return len(self.images)
    
    #now lets write a function to pick up images from the dataloader
    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.ToTensor()(im)
        
        #convert input tensor to range of -1 to 1(i.e like norm?)
        im_tensor = (2*im_tensor)-1
        return im_tensor
    
    