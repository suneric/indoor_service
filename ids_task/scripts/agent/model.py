import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
force-vision fusion actor network
"""
def fv_actor_network(image_shape,force_dim,output_dim):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3), padding='same', activation='relu')(v_input)
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
