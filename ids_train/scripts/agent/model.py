import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(keras.layers.Layer):
    """Use (mean,log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        mean, logv = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch,dim))
        return mean + tf.exp(0.5*logv)*eps

def conv_encoder(image_shape, latent_dim):
    input = keras.Input(shape=image_shape)
    h = keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(input)
    h = keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=2,padding='same',activation='relu')(h)
    h = keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=2,padding='same',activation='relu')(h)
    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(32,activation='relu')(h)
    z_mean = keras.layers.Dense(latent_dim, name='z_mean')(h)
    z_logv = keras.layers.Dense(latent_dim,name='z_logv')(h)
    z = Sampling()([z_mean,z_logv])
    model = keras.Model(input,[z_mean,z_logv,z],name='encoder')
    print(model.summary())
    return model

def conv_decoder(latent_dim):
    input = keras.Input(shape=(latent_dim,))
    h = keras.layers.Dense(8*8*32, activation='relu')(input)
    h = keras.layers.Reshape((8,8,32))(h)
    h = keras.layers.Conv2DTranspose(filters=8,kernel_size=3,strides=2,padding='same',activation='relu')(h)
    h = keras.layers.Conv2DTranspose(filters=16,kernel_size=3,strides=2,padding='same',activation='relu')(h)
    h = keras.layers.Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same',activation='relu')(h)
    output = keras.layers.Conv2DTranspose(filters=1,kernel_size=3,padding='same',activation='sigmoid')(h)
    model = keras.Model(input,output,name='decoder')
    print(model.summary())
    return model


def actor_network(image_shape,force_dim,output_dim,activation,output_activation,seq_len=None):
    vision_out, force_out = None, None
    if seq_len is None:
        # CNN-like vision network
        vision_input = keras.Input(shape=image_shape)
        vh = layers.Conv2D(32,(3,3), padding='same', activation=activation)(vision_input)
        vh = layers.MaxPool2D((2,2))(vh)
        vh = layers.Conv2D(16, (3,3), padding='same', activation=activation)(vh)
        vh = layers.MaxPool2D((2,2))(vh)
        vh = layers.Conv2D(8, (3,3), padding='same', activation=activation)(vh)
        vh = layers.Flatten()(vh)
        vision_out = layers.Dense(64, activation=activation)(vh)
        # MLP force network
        force_input = keras.Input(shape=(force_dim))
        fh = layers.Dense(32, activation=activation)(force_input)
        force_out = layers.Dense(16, activation=activation)(fh)
    else:
        vision_input = keras.Input(shape=tuple([seq_len]+list(image_shape)))
        vh = layers.ConvLSTM2D(16,kernel_size=(3,3),padding='same',activation=activation,return_sequences=True)(vision_input)
        vh = layers.MaxPool3D((1,2,2))(vh)
        vh = layers.ConvLSTM2D(16,kernel_size=(3,3),padding='same',activation=activation,return_sequences=True)(vh)
        vh = layers.MaxPool3D((1,2,2))(vh)
        vh = layers.ConvLSTM2D(8,kernel_size=(3,3),padding='same',activation=activation,return_sequences=False)(vh)
        vh = layers.Flatten()(vh)
        vision_out = layers.Dense(64, activation=activation)(vh)

        force_input = keras.Input(shape=(seq_len,force_dim))
        fh = layers.LSTM(32,activation=activation,return_sequences=False)(force_input)
        force_out = layers.Dense(16, activation=activation)(fh)
    # concatenate vision and force
    concat = layers.concatenate([vision_out, force_out])
    output = layers.Dense(32, activation=activation)(concat)
    last_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    output = layers.Dense(output_dim, activation=output_activation,kernel_initializer=last_init)(output)
    model = keras.Model(inputs=[vision_input, force_input], outputs=output)
    print(model.summary())
    return model

def critic_network(image_shape,force_dim,activation,seq_len=None):
    if seq_len is None:
        # CNN-like vision network
        vision_input = keras.Input(shape=image_shape)
        vh = layers.Conv2D(32,(3,3), padding='same', activation=activation)(vision_input)
        vh = layers.MaxPool2D((2,2))(vh)
        vh = layers.Conv2D(16, (3,3), padding='same', activation=activation)(vh)
        vh = layers.MaxPool2D((2,2))(vh)
        vh = layers.Conv2D(8, (3,3), padding='same', activation=activation)(vh)
        vh = layers.Flatten()(vh)
        vision_out = layers.Dense(64, activation=activation)(vh)
        # MLP force network
        force_input = keras.Input(shape=(force_dim))
        fh = layers.Dense(32, activation=activation)(force_input)
        force_out = layers.Dense(16, activation=activation)(fh)
    else:
        vision_input = keras.Input(shape=tuple([seq_len]+list(image_shape)))
        vh = layers.ConvLSTM2D(16,kernel_size=(3,3),padding='same',activation=activation,return_sequences=True)(vision_input)
        vh = layers.MaxPool3D((1,2,2))(vh)
        vh = layers.ConvLSTM2D(16,kernel_size=(3,3),padding='same',activation=activation,return_sequences=True)(vh)
        vh = layers.MaxPool3D((1,2,2))(vh)
        vh = layers.ConvLSTM2D(8,kernel_size=(3,3),padding='same',activation=activation,return_sequences=False)(vh)
        vh = layers.Flatten()(vh)
        vision_out = layers.Dense(64, activation=activation)(vh)

        force_input = keras.Input(shape=(seq_len,force_dim))
        fh = layers.LSTM(32,activation=activation,return_sequences=False)(force_input)
        force_out = layers.Dense(16, activation=activation)(fh)
    # concatenate vision and force
    concat = layers.concatenate([vision_out, force_out])
    output = layers.Dense(32, activation=activation)(concat)
    output = layers.Dense(1, activation='linear')(output)
    model = keras.Model(inputs=[vision_input, force_input], outputs=output)
    print(model.summary())
    return model
