# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019

@author: Reza Azad
"""
from __future__ import division

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import tensorflow as tf

#function to obtain data for training/testing (validation)
from extract_patches import *
from extract_patches import get_data_training
from help_functions import *

tf.random.set_seed(42)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--patch_height', type=int, default=64, required=False)
parser.add_argument('--patch_width', type=int, default=64, required=False)
parser.add_argument('--N_subimgs', type=int, default=20000, required=False)
args = parser.parse_args()



#========= Load settings from Config file
#patch to the datasets
path_data = f'./data/{args.dataset}_datasets_training_testing/'


print('extracting patches')
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + f'{args.dataset}_dataset_imgs_train.hdf5',
    DRIVE_train_groudTruth    = path_data + f'{args.dataset}_dataset_groundTruth_train.hdf5',  #masks
    patch_height = args.patch_height,
    patch_width  = args.patch_width,
    N_subimgs    = args.N_subimgs,
    inside_FOV = 'True', #select the patches only inside the FOV  (default == True),
    crop_train_imgs=False
)


np.save(f'{args.dataset}_patches_imgs_train',patches_imgs_train)
np.save(f'{args.dataset}_patches_masks_train',patches_masks_train)