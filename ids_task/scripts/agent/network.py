import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

class Sampling(layers.Layer):
    """
    Use (mean,log_var) to sample z, the vector encoding a digit.
    log_var = log(sigma^2) = 2*log(sigma)
    sample = sigma*x + mean, where x belongs to normal distribution N(0,1)
    """
    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch,dim)) # noise
        return mean + tf.exp(0.5*log_var) * epsilon

class DistLayer(layers.Layer):
    def call(self, inputs):
        mean = inputs
        dist = tfd.Normal(mean,1.0)
        return tfd.Independent(dist,reinterpreted_batch_ndims=1)

"""
force-vision fusion encoder
"""
def fv_encoder(image_shape, force_dim, latent_dim):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(v_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(filters=16,kernel_size=(3,3),strides=2,padding='same',activation='relu')(vh)
    vh = layers.Conv2D(filters=8,kernel_size=(3,3),strides=2,padding='same',activation='relu')(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32,activation='relu')(vh)

    f_input = keras.Input(shape=(force_dim))
    fh = layers.Dense(32, activation='relu')(f_input)
    f_output = layers.Dense(16, activation='relu')(fh)

    h = layers.concatenate([v_output, f_output])
    h = layers.Dense(32,activation='relu')(h)
    mu = layers.Dense(latent_dim, name='z_mean')(h)
    log_var = layers.Dense(latent_dim,name='z_log_var')(h)
    z = Sampling()([mu,log_var])
    model = keras.Model(inputs=[v_input,f_input],outputs=[mu,log_var,z],name='encoder')
    print(model.summary())
    return model

"""
force-vision fusion decoder
"""
def fv_decoder(latent_dim):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(32,activation='relu')(z_input)
    h = layers.Dense(4*4*8+16, activation='relu')(h)
    vh = layers.Lambda(lambda x: x[:,0:4*4*8])(h) # split layer
    vh = layers.Reshape((4,4,8))(vh)
    vh = layers.Conv2DTranspose(filters=8,kernel_size=3,strides=2,padding='same',activation='relu')(vh)
    vh = layers.Conv2DTranspose(filters=16,kernel_size=3,strides=2,padding='same',activation='relu')(vh)
    vh = layers.UpSampling2D((2,2))(vh)
    vh = layers.Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same',activation='relu')(vh)
    v_output = layers.Conv2DTranspose(filters=1,kernel_size=3,padding='same',activation='sigmoid')(vh)

    fh = layers.Lambda(lambda x: x[:,4*4*8:])(h) # split layer
    fh = layers.Dense(32,activation='relu')(fh)
    f_output = layers.Dense(3, activation='tanh')(fh)

    model = keras.Model(inputs=z_input,outputs=[v_output,f_output],name='decoder')
    print(model.summary())
    return model

"""
z dynamics model z_t | z_{t-1}, a_{t-1}
output z1 distribution mean and log variance
"""
def latent_dynamics_network(latent_dim,action_dim):
    z_input = keras.Input(shape=(latent_dim,))
    a_input = keras.Input(shape=(action_dim,))
    concat = layers.concatenate([z_input,a_input])
    h = layers.Dense(32,activation='relu')(concat)
    h = layers.Dense(32,activation='relu')(h)
    mean = layers.Dense(latent_dim, name='z1_mu')(h)
    logv = layers.Dense(latent_dim,name='z1_log_var')(h)
    z1 = Sampling()([mean,logv])
    model = keras.Model(inputs=[z_input,a_input],outputs=[mean,logv,z1],name='latent_forward_dynamics')
    print(model.summary())
    return model


"""
z actor network
"""
def latent_actor_network(latent_dim, output_dim):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(32, activation='relu')(z_input)
    h = layers.Dense(32, activation='relu')(h)
    output = layers.Dense(output_dim, activation='linear')(h)
    model = keras.Model(inputs=z_input,outputs=output)
    print(model.summary())
    return model

"""
z critic network
"""
def latent_critic_network(latent_dim):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(32, activation='relu')(z_input)
    h = layers.Dense(32, activation='relu')(h)
    output = layers.Dense(1,activation='linear')(h)
    model = keras.Model(inputs=z_input,outputs=output)
    print(model.summary())
    return model

"""
latent reward
"""
def latent_reward_network(latent_dim):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(32, activation='relu')(z_input)
    h = layers.Dense(32, activation='relu')(h)
    output = layers.Dense(1,activation='linear')(h)
    model = keras.Model(inputs=z_input,outputs=output,name='latent_reward')
    print(model.summary())
    return model
