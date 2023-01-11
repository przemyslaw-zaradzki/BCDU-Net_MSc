# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019

@author: Reza winchester
"""
from __future__ import division
from ast import arg
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import models as M
import numpy as np
from help_functions import *
from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau
from keras import callbacks

import datetime
import tensorflow as tf
tf.random.set_seed(42)



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--name_experiment', type=str, default="test", required=False)
parser.add_argument('--epochs', type=int, default=25, required=False)
parser.add_argument('--batch_size', type=int, default=8, required=False)
args = parser.parse_args()



log_dir = f"logs/fit/{args.dataset}_loss" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


#========= Load settings from Config file
#patch to the datasets
path_data = f'./data/{args.dataset}_datasets_training_testing/'
#Experiment name
name_experiment = args.name_experiment
#training settings

batch_size = args.batch_size

####################################  Load Data #####################################3
patches_imgs_train  = np.load(f'{args.dataset}_patches_imgs_train.npy')
patches_masks_train = np.load(f'{args.dataset}_patches_masks_train.npy')

patches_imgs_train = np.einsum('klij->kijl', patches_imgs_train)
patches_masks_train = np.einsum('klij->kijl', patches_masks_train)

print('Patch extracted')

#model = M.unet2_segment(input_size = (64,64,1))
model = M.BCDU_net_D3(input_size = (64,64,1))
model.summary()

print('Training')

nb_epoch = args.epochs

mcp_save = ModelCheckpoint(f'{args.dataset}_loss_lstm.hdf5', save_best_only=True, verbose=1, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-3, mode='min')

history = model.fit(patches_imgs_train,patches_masks_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_split=0.2, callbacks=[mcp_save, reduce_lr_loss, tensorboard_callback] )