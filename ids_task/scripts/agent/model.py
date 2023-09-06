import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .util import Sampling

"""
vision force [joint] actor network
"""
def actor_network(image_shape,force_dim,output_dim,joint_dim=None,act='relu',out_act='linear',out_limit=None,maxpool=False):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3),padding='same',activation=act)(v_input)
    if maxpool:
        vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(16,(3,3),padding='same',activation=act)(vh)
    if maxpool:
        vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(8,(3,3),padding='same',activation=act)(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32,activation=act)(vh)

    f_input = keras.Input(shape=(force_dim,))
    fh = layers.Dense(32,activation=act)(f_input)
    f_output = layers.Dense(16,activation=act)(fh)

    if joint_dim is not None:
        j_input = keras.Input(shape=(joint_dim,))
        jh = layers.Dense(16,activation=act)(j_input)
        j_output = layers.Dense(8,activation=act)(jh)

        concat = layers.concatenate([v_output,f_output,j_output])
        output = layers.Dense(128,activation=act)(concat)
        last_init = tf.random_uniform_initializer(minval=-3e-3,maxval=3e-3)
        output = layers.Dense(output_dim, activation=out_act,kernel_initializer=last_init)(output)
        if out_limit is not None:
            output = output*out_limit
        return keras.Model(inputs=[v_input,f_input,j_input],outputs=output,name='vfj_actor')
    else:
        concat = layers.concatenate([v_output,f_output])
        output = layers.Dense(128,activation='relu')(concat)
        last_init = tf.random_uniform_initializer(minval=-3e-3,maxval=3e-3)
        output = layers.Dense(output_dim,activation=out_act,kernel_initializer=last_init)(output)
        if out_limit is not None:
            output = output*out_limit
        return keras.Model(inputs=[v_input,f_input],outputs=output,name='vf_actor')

"""
vision force [joint] critic network
"""
def critic_network(image_shape,force_dim,joint_dim=None,act='relu',out_act='linear',maxpool=False):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3),padding='same',activation=act)(v_input)
    if maxpool:
        vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(16,(3,3),padding='same',activation=act)(vh)
    if maxpool:
        vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(8,(3,3),padding='same',activation=act)(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32,activation=act)(vh)

    f_input = keras.Input(shape=(force_dim,))
    fh = layers.Dense(32,activation=act)(f_input)
    f_output = layers.Dense(16,activation=act)(fh)

    if joint_dim is not None:
        j_input = keras.Input(shape=(joint_dim,))
        jh = layers.Dense(16,activation=act)(j_input)
        j_output = layers.Dense(8,activation=act)(jh)

        concat = layers.concatenate([v_output,f_output,j_output])
        output = layers.Dense(128,activation=act)(concat)
        output = layers.Dense(1,activation=out_act)(output)
        return keras.Model(inputs=[v_input,f_input,j_input],outputs=output,name='vfj_critic')
    else:
        concat = layers.concatenate([v_output,f_output])
        output = layers.Dense(128,activation=act)(concat)
        output = layers.Dense(1,activation=out_act)(output)
        return keras.Model(inputs=[v_input,f_input],outputs=output,name='vf_critic')

"""
vision force [joint] twin critic network
"""
def twin_critic_network(image_shape,force_dim,action_dim,joint_dim=None,act='relu',out_act='linear',maxpool=False):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3),padding='same',activation=act)(v_input)
    if maxpool:
        vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(16,(3,3),padding='same',activation=act)(vh)
    if maxpool:
        vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(8,(3,3),padding='same',activation=act)(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32,activation=act)(vh)

    f_input = keras.Input(shape=(force_dim,))
    fh = layers.Dense(32,activation=act)(f_input)
    f_output = layers.Dense(16,activation=act)(fh)

    a_input = keras.Input(shape=(action_dim,))
    ah = layers.Dense(16, activation=act)(a_input)
    a_output = layers.Dense(8, activation=act)(ah)

    if joint_dim is not None:
        j_input = keras.Input(shape=(joint_dim,))
        jh = layers.Dense(16,activation=act)(j_input)
        j_output = layers.Dense(8,activation=act)(jh)

        concat = layers.concatenate([v_output,f_output,j_output,a_output])
        output_1 = layers.Dense(32, activation=act)(concat)
        output_1 = layers.Dense(1, activation=out_act)(output_1)
        output_2 = layers.Dense(32, activation=act)(concat)
        output_2 = layers.Dense(1, activation=out_act)(output_2)
        return keras.Model(inputs=[v_input,f_input,j_input,a_input], outputs=[output_1,output_2], name='vfj_twin_critic')
    else:
        concat = layers.concatenate([v_output,f_output,a_output])
        output_1 = layers.Dense(32, activation=act)(concat)
        output_1 = layers.Dense(1, activation=out_act)(output_1)
        output_2 = layers.Dense(32, activation=act)(concat)
        output_2 = layers.Dense(1, activation=out_act)(output_2)
        return keras.Model(inputs=[v_input,f_input,a_input], outputs=[output_1,output_2], name='vf_twin_critic')

"""
observation encoder
"""
def obs_encoder(image_shape,force_dim,latent_dim,act='relu'):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation=act)(v_input)
    vh = layers.Conv2D(filters=16,kernel_size=(3,3),strides=2,padding='same',activation=act)(vh)
    vh = layers.Conv2D(filters=8,kernel_size=(3,3),strides=2,padding='same',activation=act)(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32,activation=act)(vh)

    f_input = keras.Input(shape=(force_dim))
    fh = layers.Dense(32, activation=act)(f_input)
    f_output = layers.Dense(16, activation=act)(fh)

    h = layers.concatenate([v_output, f_output])
    h = layers.Dense(128,activation=act)(h)
    mu = layers.Dense(latent_dim,name="z_mean")(h)
    logv = layers.Dense(latent_dim,name="z_logv")(h)
    z = Sampling()([mu,logv])
    model = keras.Model(inputs=[v_input,f_input],outputs=[mu,logv,z],name='obs_encoder')
    # print(model.summary())
    return model

"""
observation decoder
"""
def obs_decoder(latent_dim,act='elu',scale=0.5):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(32,activation=act)(z_input)
    h = layers.Dense(512+16, activation=act)(h)
    vh = layers.Lambda(lambda x: x[:,0:512])(h) # split layer
    vh = layers.Reshape((8,8,8))(vh)
    vh = layers.Conv2DTranspose(filters=8,kernel_size=3,strides=2,padding='same',activation=act)(vh)
    vh = layers.Conv2DTranspose(filters=16,kernel_size=3,strides=2,padding='same',activation=act)(vh)
    vh = layers.Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same',activation=act)(vh)
    v_output = layers.Conv2DTranspose(filters=1,kernel_size=3,padding='same',activation='tanh')(vh)
    v_output = scale*v_output

    fh = layers.Lambda(lambda x: x[:,512:])(h) # split layer
    fh = layers.Dense(32,activation=act)(fh)
    f_output = layers.Dense(3, activation='tanh')(fh)
    model = keras.Model(inputs=z_input,outputs=[v_output,f_output],name='obs_decoder')
    # print(model.summary())
    return model

"""
z actor network
"""
def latent_actor(latent_dim,output_dim,act='relu'):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(32, activation=act)(z_input)
    h = layers.Dense(32, activation=act)(h)
    output = layers.Dense(output_dim,activation='linear')(h)
    model = keras.Model(inputs=z_input,outputs=output,name='latent_actor')
    # print(model.summary())
    return model

def latent_recurrent_actor(latent_dim,output_dim,seq_len,act='relu'):
    z_input = keras.Input(shape=(seq_len,latent_dim))
    h = layers.LSTM(32,activation=act,return_sequences=False)(z_input)
    h = layers.Dense(32,activation=act)(h)
    output = layers.Dense(output_dim,activation='linear')(h)
    model = keras.Model(inputs=z_input,outputs=output,name='latent_recurrent_actor')
    # print(model.summary())
    return model

"""
z critic network
"""
def latent_critic(latent_dim,act='relu'):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(32, activation=act)(z_input)
    h = layers.Dense(32, activation=act)(h)
    output = layers.Dense(1,activation='linear')(h)
    model = keras.Model(inputs=z_input,outputs=output,name='latent_critic')
    # print(model.summary())
    return model

def latent_recurrent_critic(latent_dim,seq_len,act='relu'):
    z_input = keras.Input(shape=(seq_len,latent_dim))
    h = layers.LSTM(32,activation=act,return_sequences=False)(z_input)
    h = layers.Dense(32,activation=act)(h)
    output = layers.Dense(1,activation='linear')(h)
    model = keras.Model(inputs=z_input,outputs=output,name='latent_recurrent_critic')
    # print(model.summary())
    return model

"""
z dynamics model z_t | z_{t-1}, a_{t-1}
output z1 distribution mean and log variance
"""
def latent_dynamics(latent_dim,action_dim,act='elu',out_act='linear'):
    z_input = keras.Input(shape=(latent_dim,))
    a_input = keras.Input(shape=(action_dim,))
    concat = layers.concatenate([z_input,a_input])
    h = layers.Dense(64,activation=act,kernel_initializer='random_normal')(concat)
    h = layers.Dense(64,activation=act,kernel_initializer='random_normal')(h)
    mu = layers.Dense(latent_dim,activation=out_act)(h)
    sigma = layers.Dense(latent_dim,activation=out_act)(h)
    model = keras.Model(inputs=[z_input,a_input],outputs=[mu,sigma],name='latent_dynamics')
    # print(model.summary())
    return model

"""
latent reward
"""
def latent_reward(latent_dim,act='relu',out_act='softmax',output_unit=10):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(32, activation=act,kernel_initializer='random_normal')(z_input)
    h = layers.Dense(32, activation=act,kernel_initializer='random_normal')(h)
    output = layers.Dense(output_unit,activation=out_act)(h)
    model = keras.Model(inputs=z_input,outputs=output,name='latent_reward_classifier')
    # print(model.summary())
    return model

"""
vision encoder
"""
def vision_encoder(image_shape,latent_dim,act='relu'):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation=act)(v_input)
    vh = layers.Conv2D(filters=16,kernel_size=(3,3),strides=2,padding='same',activation=act)(vh)
    vh = layers.Conv2D(filters=8,kernel_size=(3,3),strides=2,padding='same',activation=act)(vh)
    vh = layers.Flatten()(vh)
    vh = layers.Dense(32,activation=act)(vh)
    mu = layers.Dense(latent_dim,name="z_mean")(vh)
    logv = layers.Dense(latent_dim,name="z_logv")(vh)
    z = Sampling()([mu,logv])
    model = keras.Model(inputs=v_input,outputs=[mu,logv,z],name='vision_encoder')
    print(model.summary())
    return model

"""
vision decoder
"""
def vision_decoder(latent_dim,act='elu',scale=0.5):
    z_input = keras.Input(shape=(latent_dim,))
    vh = layers.Dense(32,activation=act)(z_input)
    vh = layers.Dense(512, activation=act)(vh)
    vh = layers.Reshape((8,8,8))(vh)
    vh = layers.Conv2DTranspose(filters=8,kernel_size=3,strides=2,padding='same',activation=act)(vh)
    vh = layers.Conv2DTranspose(filters=16,kernel_size=3,strides=2,padding='same',activation=act)(vh)
    vh = layers.Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same',activation=act)(vh)
    v_output = layers.Conv2DTranspose(filters=1,kernel_size=3,padding='same',activation='tanh')(vh)
    v_output = scale*v_output # output in [-0.5,0.5]
    model = keras.Model(inputs=z_input,outputs=v_output,name='vision_decoder')
    print(model.summary())
    return model

"""
z + force actor network
"""
def latent_force_actor(latent_dim,force_dim,output_dim,act='relu'):
    z_input = keras.Input(shape=(latent_dim,))
    f_input = keras.Input(shape=(force_dim,))
    input = layers.concatenate([z_input, f_input])
    h = layers.Dense(32, activation=act)(input)
    h = layers.Dense(32, activation=act)(h)
    output = layers.Dense(output_dim,activation='linear')(h)
    model = keras.Model(inputs=[z_input,f_input],outputs=output,name='latent_force_actor')
    # print(model.summary())
    return model

"""
z + force critic network
"""
def latent_force_critic(latent_dim,force_dim,act='relu'):
    z_input = keras.Input(shape=(latent_dim,))
    f_input = keras.Input(shape=(force_dim,))
    input = layers.concatenate([z_input, f_input])
    h = layers.Dense(32, activation=act)(input)
    h = layers.Dense(32, activation=act)(h)
    output = layers.Dense(1,activation='linear')(h)
    model = keras.Model(inputs=[z_input,f_input],outputs=output,name='latent_critic')
    # print(model.summary())
    return model


def fv_encoder(image_shape,force_dim,latent_dim,act='relu'):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(v_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(filters=16,kernel_size=(3,3),strides=2,padding='same',activation='relu')(vh)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(filters=8,kernel_size=(3,3),strides=2,padding='same',activation='relu')(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32,activation='relu')(vh)

    f_input = keras.Input(shape=(force_dim))
    fh = layers.Dense(32, activation='relu')(f_input)
    f_output = layers.Dense(16, activation='relu')(fh)

    h = layers.concatenate([v_output, f_output])
    h = layers.Dense(128,activation='relu')(h)
    mean = layers.Dense(latent_dim, name='z_mean')(h)
    logv = layers.Dense(latent_dim,name='z_log_var')(h)
    z = Sampling()([mean,logv])
    model = keras.Model(inputs=[v_input,f_input],outputs=[mean,logv,z],name='encoder')
    # print(model.summary())
    return model

"""
force-vision fusion decoder
"""
def fv_decoder(latent_dim,scale=0.5):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(32,activation='relu')(z_input)
    h = layers.Dense(32+16, activation='relu')(h)
    vh = layers.Lambda(lambda x: x[:,0:32])(h) # split layer
    vh = layers.Reshape((2,2,8))(vh)
    vh = layers.Conv2DTranspose(filters=8,kernel_size=3,strides=2,padding='same',activation='relu')(vh)
    vh = layers.UpSampling2D((2,2))(vh)
    vh = layers.Conv2DTranspose(filters=16,kernel_size=3,strides=2,padding='same',activation='relu')(vh)
    vh = layers.UpSampling2D((2,2))(vh)
    vh = layers.Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same',activation='relu')(vh)
    v_output = layers.Conv2DTranspose(filters=1,kernel_size=3,padding='same',activation='tanh')(vh)
    v_output = scale*v_output # output in [-0.5,0.5]

    fh = layers.Lambda(lambda x: x[:,32:])(h) # split layer
    fh = layers.Dense(32,activation='relu')(fh)
    f_output = layers.Dense(3, activation='tanh')(fh)

    model = keras.Model(inputs=z_input,outputs=[v_output,f_output],name='decoder')
    # print(model.summary())
    return model
