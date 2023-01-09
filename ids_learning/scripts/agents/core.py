import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal

def discount_cumsum(x,discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors
    input: vector x: [x0, x1, x2]
    output: [x0+discount*x1+discount^2*x2, x1+discount*x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def copy_network_variables(target_weights, from_weights, polyak = 0.0):
    """
    copy network variables with consider a polyak
    In DQN-based algorithms, the target network is just copied over from the main network
    every some-fixed-number of steps. In DDPG-style algorithm, the target network is updated
    once per main network update by polyak averaging, where polyak(tau) usually close to 1.
    """
    for (a,b) in zip(target_weights, from_weights):
        a.assign(a*polyak + b*(1-polyak))

def force_actor_network(force_dim,output_dim,activation,output_activation,output_limit=None):
    # MLP force network
    force_input = keras.Input(shape=(force_dim))
    fh = layers.Dense(32, activation=activation)(force_input)
    fh = layers.Dense(16, activation=activation)(fh)
    last_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    output = layers.Dense(output_dim, activation=output_activation,kernel_initializer=last_init)(fh)
    if output_limit is not None:
        output = output * output_limit
    model = keras.Model(inputs=force_input, outputs=output)
    return model

def vision_force_actor_network(image_shape,force_dim,output_dim,activation,output_activation,output_limit=None):
    # CNN-like vision network
    vision_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3), padding='same', activation=activation)(vision_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(32, (3,3), padding='same', activation=activation)(vh)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(32, (3,3), padding='same', activation=activation)(vh)
    vh = layers.Flatten()(vh)
    vision_out = layers.Dense(128, activation=activation)(vh)
    # MLP force network
    force_input = keras.Input(shape=(force_dim))
    fh = layers.Dense(32, activation=activation)(force_input)
    force_out = layers.Dense(16, activation=activation)(fh)
    # concatenate vision and force
    concat = layers.concatenate([vision_out, force_out])
    output = layers.Dense(32, activation=activation)(concat)
    last_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    output = layers.Dense(output_dim, activation=output_activation,kernel_initializer=last_init)(output)
    if output_limit is not None:
        output = output * output_limit
    model = keras.Model(inputs=[vision_input, force_input], outputs=output)
    return model

def vision_force_guassian_actor_network(image_shape, force_dim, output_dim, activation):
    vision_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3), padding='same', activation=activation)(vision_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(32, (3,3), padding='same', activation=activation)(vh)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(32, (3,3), padding='same', activation=activation)(vh)
    vh = layers.Flatten()(vh)
    vision_out = layers.Dense(128, activation=activation)(vh)
    # MLP force network
    force_input = keras.Input(shape=(force_dim))
    fh = layers.Dense(32, activation=activation)(force_input)
    force_out = layers.Dense(16, activation=activation)(fh)
    # concatenate vision and force
    concat = layers.concatenate([vision_out, force_out])
    output = layers.Dense(32, activation=activation)(concat)
    mu = layers.Dense(output_dim)(output)
    logstd = layers.Dense(output_dim)(output)
    model = keras.Model(inputs=[vision_input, force_input], outputs=[mu,logstd])
    return model

def vision_force_critic_network(image_shape, force_dim, activation):
    # CNN-like vision network
    vision_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3), padding='same', activation=activation)(vision_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(32, (3,3), padding='same', activation=activation)(vh)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(32, (3,3), padding='same', activation=activation)(vh)
    vh = layers.Flatten()(vh)
    vision_out = layers.Dense(128, activation=activation)(vh)
    # MLP force network
    force_input = keras.Input(shape=(force_dim))
    fh = layers.Dense(32, activation=activation)(force_input)
    force_out = layers.Dense(16, activation=activation)(fh)
    # concatenate vision and force
    concat = layers.concatenate([vision_out, force_out])
    output = layers.Dense(32, activation=activation)(concat)
    output = layers.Dense(1, activation='linear')(output)
    model = keras.Model(inputs=[vision_input, force_input], outputs=output)
    return model

def vision_force_action_twin_critic_network(image_shape, force_dim, action_dim, activation):
    # CNN-like vision network
    vision_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3), padding='same', activation=activation)(vision_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(32, (3,3), padding='same', activation=activation)(vh)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(32, (3,3), padding='same', activation=activation)(vh)
    vh = layers.Flatten()(vh)
    vision_out = layers.Dense(128, activation=activation)(vh)
    # MLP force network
    force_input = keras.Input(shape=(force_dim))
    fh = layers.Dense(32, activation=activation)(force_input)
    force_out = layers.Dense(16, activation=activation)(fh)
    # action network
    action_input = keras.Input(shape=(action_dim))
    ah = layers.Dense(16, activation=activation)(action_input)
    action_out = layers.Dense(8, activation=activation)(ah)
    # concatenate vision and force
    concat = layers.concatenate([vision_out, force_out, action_out])
    output_1 = layers.Dense(32, activation=activation)(concat)
    output_1 = layers.Dense(1, activation='linear')(output_1)
    output_2 = layers.Dense(32, activation=activation)(concat)
    output_2 = layers.Dense(1, activation='linear')(output_2)
    model = keras.Model(inputs=[vision_input, force_input, action_input], outputs=[output_1, output_2])
    return model

def vision_force_joint_actor_network(image_shape,force_dim,joint_dim,output_dim,activation,output_activation,output_limit=None):
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
    output = layers.Dense(32, activation=activation)(concat)
    last_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    output = layers.Dense(output_dim, activation=output_activation,kernel_initializer=last_init)(output)
    if output_limit is not None:
        output = output * output_limit
    model = keras.Model(inputs=[vision_input,force_input,joint_input], outputs=output)
    return model

def vision_force_joint_critic_network(image_shape, force_dim, joint_dim, activation):
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
    output = layers.Dense(32, activation=activation)(concat)
    output = layers.Dense(1, activation='linear')(output)
    model = keras.Model(inputs=[vision_input, force_input, joint_input], outputs=output)
    return model
