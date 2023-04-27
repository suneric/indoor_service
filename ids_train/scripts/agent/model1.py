import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
PPO policy for input combined with images, forces, and joints.
"""

def actor_network(image_shape,force_dim,joint_dim,output_dim,activation,output_activation,output_limit=None):
    vision_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3), padding='same', activation=activation)(vision_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(16, (3,3), padding='same', activation=activation)(vh)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(8, (3,3), padding='same', activation=activation)(vh)
    vh = layers.Flatten()(vh)
    vision_out = layers.Dense(32, activation=activation)(vh)

    force_input = keras.Input(shape=(force_dim))
    fh = layers.Dense(32, activation=activation)(force_input)
    force_out = layers.Dense(16, activation=activation)(fh)

    joint_input = keras.Input(shape=(joint_dim))
    jh = layers.Dense(16, activation=activation)(joint_input)
    joint_out = layers.Dense(8, activation=activation)(jh)

    concat = layers.concatenate([vision_out, force_out, joint_out])
    output = layers.Dense(128, activation=activation)(concat)
    last_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    output = layers.Dense(output_dim, activation=output_activation,kernel_initializer=last_init)(output)
    if output_limit is not None:
        output = output * output_limit
    model = keras.Model(inputs=[vision_input,force_input,joint_input], outputs=output)
    print(model.summary())
    return model

def critic_network(image_shape, force_dim, joint_dim, activation):
    vision_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3), padding='same', activation=activation)(vision_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(16, (3,3), padding='same', activation=activation)(vh)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(8, (3,3), padding='same', activation=activation)(vh)
    vh = layers.Flatten()(vh)
    vision_out = layers.Dense(32, activation=activation)(vh)

    force_input = keras.Input(shape=(force_dim))
    fh = layers.Dense(32, activation=activation)(force_input)
    force_out = layers.Dense(16, activation=activation)(fh)

    joint_input = keras.Input(shape=(joint_dim))
    jh = layers.Dense(16, activation=activation)(joint_input)
    joint_out = layers.Dense(8, activation=activation)(jh)

    concat = layers.concatenate([vision_out, force_out, joint_out])
    output = layers.Dense(128, activation=activation)(concat)
    output = layers.Dense(1, activation='linear')(output)
    model = keras.Model(inputs=[vision_input, force_input, joint_input], outputs=output)
    print(model.summary())
    return model
