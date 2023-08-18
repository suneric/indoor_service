#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from agent.gan import CycleGAN
import matplotlib.pyplot as plt
from utility import *
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--size', type=int, default=None)
    parser.add_argument('--iter', type=int, default=300)
    return parser.parse_args()

adv_loss_fn = keras.losses.MeanSquaredError()

def generator_loss_fn(fake):
    return adv_loss_fn(tf.ones_like(fake),fake)

def discriminator_loss_fn(real,fake):
    real_loss = adv_loss_fn(tf.ones_like(real),real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake),fake)
    return (real_loss+fake_loss)/2

if __name__=="__main__":
    args = get_args()
    data_dir = os.path.join(sys.path[0],"../../dump/collection/")
    train_data = load_observation(os.path.join(data_dir,"env"))
    test_data = load_observation(os.path.join(data_dir,args.env))
    train_images, all_images = train_data['image'], test_data['image']
    train_images = train_images[:args.size]
    test_images, validate_images = all_images[:args.size], all_images[args.size:]

    print("=== i2i traning,{},{},{}".format(len(train_images),len(test_images),len(validate_images)))

    model = CycleGAN(image_shape=(64,64,1))
    model.compile(
        gen_G_opt = keras.optimizers.Adam(learning_rate=2e-4,beta_1=0.5),
        gen_F_opt = keras.optimizers.Adam(learning_rate=2e-4,beta_1=0.5),
        disc_X_opt = keras.optimizers.Adam(learning_rate=2e-4,beta_1=0.5),
        disc_Y_opt = keras.optimizers.Adam(learning_rate=2e-4,beta_1=0.5),
        gen_loss_fn = generator_loss_fn,
        disc_loss_fn = discriminator_loss_fn,
    )
    model.fit(x=test_images,y=train_images, epochs=args.iter)
    model.save(os.path.join(sys.path[0],"../../saved_models/door_open/i2i"))

    save_dir = os.path.join(data_dir,"i2i")
    for i in range(len(validate_images)):
        real = validate_images[i]
        transfer = tf.squeeze(model.gen_G(tf.expand_dims(tf.convert_to_tensor(real),0))).numpy()
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(real,cmap='gray')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].imshow(transfer,cmap='gray')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        imagePath = os.path.join(save_dir,"step{}".format(i))
        plt.savefig(imagePath)
        plt.close(fig)
