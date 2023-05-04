import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """Use (mean,log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch,dim)) # noise
        return mu + tf.exp(0.5*sigma) * epsilon

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
    z_mu = layers.Dense(latent_dim, name='z_mean')(h)
    z_sigma = layers.Dense(latent_dim,name='z_sigma')(h)
    z = Sampling()([z_mu,z_sigma])
    model = keras.Model(inputs=[v_input,f_input],outputs=[z_mu,z_sigma,z],name='encoder')
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
z actor network
"""
def latent_actor_network(latent_dim, output_dim):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(64, activation='relu')(z_input)
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
    h = layers.Dense(64, activation='relu')(z_input)
    h = layers.Dense(32, activation='relu')(h)
    output = layers.Dense(1, activation='linear')(h)
    model = keras.Model(inputs=z_input,outputs=output)
    print(model.summary())
    return model

"""
z dynamics model z_t-1, a_t-1 -> z_t, r_t
"""
def latent_dynamics_network(latent_dim,action_dim):
    z_input = keras.Input(shape=(latent_dim,))
    a_input = keras.Input(shape=(action_dim,))
    concat = layers.concatenate([z_input,a_input])
    h = layers.Dense(64,activation='relu')(concat)
    h = layers.Dense(32,activation='relu')(h)
    z1_mu = layers.Dense(latent_dim, name='z1_mu')(h)
    z1_sigma = layers.Dense(latent_dim,name='z1_sigma')(h)
    z1 = Sampling()([z1_mu,z1_sigma])
    model = keras.Model(inputs=[z_input,a_input],outputs=[z1_mu,z1_sigma,z1],name='latent_forward_dynamics')
    print(model.summary())
    return model

"""
latent reward
"""
def latent_reward_network(latent_dim):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(64, activation='relu')(z_input)
    h = layers.Dense(32, activation='relu')(h)
    output = layers.Dense(1, activation='linear')(h)
    model = keras.Model(inputs=z_input,outputs=output,name='latent_reward')
    print(model.summary())
    return model

#==============================================================================#
"""
force-vision fusion actor network
"""
def fv_actor_network(image_shape,force_dim,output_dim):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3), padding='same', activation='relu')(v_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(16, (3,3), padding='same', activation='relu')(vh)
    vh = layers.Conv2D(8, (3,3), padding='same', activation='relu')(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32, activation='relu')(vh)

    f_input = keras.Input(shape=(force_dim,))
    fh = layers.Dense(32, activation='relu')(f_input)
    f_output = layers.Dense(16, activation='relu')(fh)

    concat = layers.concatenate([v_output, f_output])
    output = layers.Dense(32, activation='relu')(concat)
    output = layers.Dense(output_dim, activation='linear')(output)
    model = keras.Model(inputs=[v_input, f_input], outputs=output,name='fv_actor')
    print(model.summary())
    return model

"""
force-vision fusion critic network
"""
def fv_critic_network(image_shape,force_dim):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3), padding='same', activation='relu')(v_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(16, (3,3), padding='same', activation='relu')(vh)
    vh = layers.Conv2D(8, (3,3), padding='same', activation='relu')(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32, activation='relu')(vh)

    f_input = keras.Input(shape=(force_dim,))
    fh = layers.Dense(32, activation='relu')(f_input)
    f_output = layers.Dense(16, activation='relu')(fh)

    concat = layers.concatenate([v_output, f_output])
    output = layers.Dense(32, activation='relu')(concat)
    output = layers.Dense(1, activation='linear')(output)
    model = keras.Model(inputs=[v_input, f_input], outputs=output,name='fv_critic')
    print(model.summary())
    return model

"""
force-vision fusion recurrent actor network
"""
def fv_recurrent_actor_network(image_shape,force_dim,output_dim,seq_len):
    v_input = keras.Input(shape=[seq_len]+list(image_shape))
    vh = layers.ConvLSTM2D(32,kernel_size=(3,3),padding='same',activation='relu',return_sequences=True)(v_input)
    vh = layers.MaxPool3D((1,2,2))(vh)
    vh = layers.ConvLSTM2D(16,kernel_size=(3,3),padding='same',activation='relu',return_sequences=True)(vh)
    vh = layers.ConvLSTM2D(8,kernel_size=(3,3),padding='same',activation='relu',return_sequences=False)(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32, activation='relu')(vh)

    f_input = keras.Input(shape=(seq_len,force_dim))
    fh = layers.LSTM(32,activation='relu',return_sequences=False)(f_input)
    f_output = layers.Dense(16, activation='relu')(fh)

    concat = layers.concatenate([v_output, f_output])
    output = layers.Dense(32, activation='relu')(concat)
    output = layers.Dense(output_dim, activation='linear')(output)
    model = keras.Model(inputs=[v_input, f_input], outputs=output,name='fv_recurrent_actor')
    print(model.summary())
    return model

"""
force-vision fusion recurrent critic network
"""
def fv_recurrent_critic_network(image_shape,force_dim,seq_len=None):
    v_input = keras.Input(shape=[seq_len]+list(image_shape))
    vh = layers.ConvLSTM2D(32,kernel_size=(3,3),padding='same',activation='relu',return_sequences=True)(v_input)
    vh = layers.MaxPool3D((1,2,2))(vh)
    vh = layers.ConvLSTM2D(16,kernel_size=(3,3),padding='same',activation='relu',return_sequences=True)(vh)
    vh = layers.ConvLSTM2D(8,kernel_size=(3,3),padding='same',activation='relu',return_sequences=False)(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32, activation='relu')(vh)

    f_input = keras.Input(shape=(seq_len,force_dim))
    fh = layers.LSTM(32,activation='relu',return_sequences=False)(f_input)
    f_output = layers.Dense(16, activation='relu')(fh)

    concat = layers.concatenate([v_output, f_output])
    output = layers.Dense(32, activation='relu')(concat)
    output = layers.Dense(1, activation='linear')(output)
    model = keras.Model(inputs=[v_input, f_input], outputs=output,name='fv_recurrent_critic')
    print(model.summary())
    return model

"""
vision force joint actor network
"""
def jfv_actor_network(image_shape,force_dim,joint_dim,output_dim):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3), padding='same', activation='relu')(v_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(16, (3,3), padding='same', activation='relu')(vh)
    vh = layers.Conv2D(8, (3,3), padding='same', activation='relu')(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32, activation='relu')(vh)

    f_input = keras.Input(shape=(force_dim,))
    fh = layers.Dense(32, activation='relu')(f_input)
    f_output = layers.Dense(16, activation='relu')(fh)

    j_input = keras.Input(shape=(joint_dim,))
    jh = layers.Dense(16, activation='relu')(j_input)
    j_output = layers.Dense(8, activation='relu')(jh)

    concat = layers.concatenate([v_output, f_output, j_output])
    output = layers.Dense(128, activation='relu')(concat)
    output = layers.Dense(output_dim, activation='linear')(output)
    model = keras.Model(inputs=[v_input,f_input,j_input], outputs=output,name='jfv_actor')
    print(model.summary())
    return model

def jfv_critic_network(image_shape, force_dim, joint_dim):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3), padding='same', activation='relu')(v_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(16, (3,3), padding='same', activation='relu')(vh)
    vh = layers.Conv2D(8, (3,3), padding='same', activation='relu')(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32, activation='relu')(vh)

    f_input = keras.Input(shape=(force_dim,))
    fh = layers.Dense(32, activation='relu')(f_input)
    f_output = layers.Dense(16, activation='relu')(fh)

    j_input = keras.Input(shape=(joint_dim,))
    jh = layers.Dense(16, activation='relu')(j_input)
    j_output = layers.Dense(8, activation='relu')(jh)

    concat = layers.concatenate([v_output, f_output, j_output])
    output = layers.Dense(128, activation='relu')(concat)
    output = layers.Dense(1, activation='linear')(output)
    model = keras.Model(inputs=[v_input, f_input, j_input], outputs=output,name='jfv_critic')
    print(model.summary())
    return model
