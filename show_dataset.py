import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import Dataset
from os import listdir
from os.path import splitext, isfile, join
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.data_loading import DenoiserDataset, load_image, unique_mask_values
from functools import partial
import numpy as np
from PIL import Image
from multiprocessing import Pool

import wandb
from evaluate import evaluate
from unet import UNet
from matplotlib import pyplot as plt

if __name__ == "__main__":
    dataset = DenoiserDataset(images_dir='../dataset/images', 
                           mask_dir='../dataset/gt',
                           mask_suffix='.npy')

    for data in dataset:
        print(str(data['img_name']).split('\\')[-1])
        if str(data['img_name']).split('\\')[-1] == "im21255.jpg":
            print("Huray")
            plt.imshow(data['mask'].numpy().transpose((1,2,0)))
            plt.show()
            break

    # data = dataset[0]

    # plt.imshow(data['mask'].numpy().transpose((1,2,0)))
    # plt.figure()
    # plt.imshow(data['image'].numpy().transpose((1,2,0)))
    # plt.show()
