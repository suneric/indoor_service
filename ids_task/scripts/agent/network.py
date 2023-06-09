import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
force-vision fusion encoder
"""
def fv_encoder(image_shape,force_dim,latent_dim,act='relu'):
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
    h = layers.Dense(128,activation='relu')(h)
    mean = layers.Dense(latent_dim, name='z_mean')(h)
    logv = layers.Dense(latent_dim,name='z_log_var')(h)
    model = keras.Model(inputs=[v_input,f_input],outputs=[mean,logv,z],name='encoder')
    #print(model.summary())
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
    #print(model.summary())
    return model

"""
z dynamics model z_t | z_{t-1}, a_{t-1}
output z1 distribution mean and log variance
"""
def latent_dynamics_network(latent_dim,action_dim,act='elu'):
    z_input = keras.Input(shape=(latent_dim,))
    a_input = keras.Input(shape=(action_dim,))
    concat = layers.concatenate([z_input,a_input])
    h = layers.Dense(64,activation=act,kernel_initializer='random_normal')(concat)
    h = layers.Dense(64,activation=act,kernel_initializer='random_normal')(h)
    mean = layers.Dense(latent_dim,name='z1_mu')(h)
    logv = layers.Dense(latent_dim,name='z1_log_var')(h)
    z1 = Sampling()([mean,logv])
    model = keras.Model(inputs=[z_input,a_input],outputs=[mean,logv,z1],name='latent_forward_dynamics')
    #print(model.summary())
    return model

"""
latent reward
"""
def latent_reward_network(latent_dim,act='elu'):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(32, activation=act)(z_input)
    h = layers.Dense(32, activation=act)(h)
    output = layers.Dense(1)(h)
    model = keras.Model(inputs=z_input,outputs=output,name='latent_reward')
    #print(model.summary())
    return model

"""
z actor network
"""
def latent_actor_network(latent_dim,output_dim,act='elu'):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(32, activation=act)(z_input)
    h = layers.Dense(32, activation=act)(h)
    output = layers.Dense(output_dim)(h)
    model = keras.Model(inputs=z_input,outputs=output)
    #print(model.summary())
    return model

"""
z critic network
"""
def latent_critic_network(latent_dim,act='elu'):
    z_input = keras.Input(shape=(latent_dim,))
    h = layers.Dense(32, activation=act)(z_input)
    h = layers.Dense(32, activation=act)(h)
    output = layers.Dense(1)(h)
    model = keras.Model(inputs=z_input,outputs=output)
    #print(model.summary())
    return model

"""
Observation(vision+force) VAE
reference
https://keras.io/examples/generative/vae/
https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
The encoder distribution q(z|x) = N(z|u(x),SIGMA(x)) where SIGMA = diag(var_1,...,var_n)
The latent prior is give by p(z) = N(0,I)
Both are multivariate Gaussians of dimension n, the KL divergence is
D_kl(q(z|x) || p(z)) = 0.5*(SUM(mu_i^2) + SUM(sigma_i^2) - SUM(log(sigma_i^2)+1))

Given mu, and log_var = log(sigma^2), then
kl_loss = 0.5*(mu^2 + exp(log_var) - log_var - 1)
"""
class ObservationVAE(keras.Model):
    def __init__(self,image_shape,force_dim,latent_dim,lr=3e-4,**kwargs):
        super().__init__(**kwargs)
        self.encoder = fv_encoder(image_shape, force_dim, latent_dim)
        self.decoder = fv_decoder(latent_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    @tf.function
    def train_step(self,data):
        x,y = data
        images,forces = x
        with tf.GradientTape() as tape:
            mu,logv,z = self.encoder([images,forces])
            r_images,r_forces = self.decoder(z) # reconstruction
            image_loss = tf.reduce_sum(keras.losses.MSE(images,r_images), axis=(1,2))
            force_loss = keras.losses.MSE(forces,r_forces)
            rc_loss = tf.reduce_mean(image_loss+force_loss)
            kl_loss = 0.5*(tf.square(mu) + tf.exp(logv) - logv - 1)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = rc_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return {"obs_loss":total_loss,"obs_reconstruction_loss":rc_loss,"obs_kl_loss":kl_loss}

"""
Latent dynamics model (z_t|z_{t-1})
https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
https://blogs.sas.com/content/iml/2020/06/01/the-kullback-leibler-divergence-between-continuous-probability-distributions.html
KL divergence between two normal distributions P = N(u1,s1^2), Q = N(u2,s2^2), where u is mean, s is sigma, is
D_kl(P || Q) = log(s2/s1) + (s1^2 + (u1-u2)^2)/(2*s2^2) - 1/2

given mu = u, log_var = log(s^2), the KL loss is
kl_loss = 0.5*(log_var2-log_var1) + (exp(log_var1) + (mu1-mu2)^2)/(2*exp(log_var2)) - 0.5
"""
class LatentDynamics(keras.Model):
    def __init__(self,latent_dim,action_dim,lr=1e-4,alpha=0.8,**kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.action_dim = action_dim
        self.transit = latent_dynamics_network(latent_dim, action_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    @tf.function
    def train_step(self,data):
        x,y = data
        z,a = x
        z1_mean,z1_logv,z1 = y # posterior distribution Q(z_t|x_t)
        with tf.GradientTape() as tape:
            z1_mean_prior,z1_logv_prior,z1_prior = self.transit([z,a]) # Prior distribution P(z_t|z_{t-1},a_{]t-1})
            kl_loss = self.compute_kl((z1_mean,z1_logv),(z1_mean_prior,z1_logv_prior))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        grads = tape.gradient(kl_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return {'dynamics_kl_loss':kl_loss}

    def forward(self,z,a):
        z = tf.expand_dims(tf.convert_to_tensor(z), 0)
        a = tf.expand_dims(tf.convert_to_tensor(a), 0)
        z1_mu,z1_log_var,z1 = self.transit([z,a])
        return tf.squeeze(z1).numpy()

    def compute_kl(self,p,q):
        """
        KL distance of distribution P from Q, measure how P is different from Q
        """
        dist1 = tfpd.Normal(loc=p[0],scale=tf.sqrt(tf.exp(p[1])))
        dist2 = tfpd.Normal(loc=q[0],scale=tf.sqrt(tf.exp(q[1])))
        return tfpd.kl_divergence(dist1,dist2)
        # mu_p,logv_p = p
        # mu_q,logv_q = q
        # return 0.5*(logv_q-logv_p)+(tf.exp(logv_p)+tf.square(mu_p-mu_q))/(2*tf.exp(logv_q))-0.5

"""
Reward Model (r_t|z_t)
"""
class RewardModel(keras.Model):
    def __init__(self,latent_dim,lr=1e-4,**kwargs):
        super().__init__(**kwargs)
        self.reward = latent_reward_network(latent_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    @tf.function
    def train_step(self,data):
        z,r = data
        with tf.GradientTape() as tape:
            logits = self.reward(z)
            loss = -tf.reduce_mean(tfpd.Normal(loc=logits,scale=1.0).log_prob(r))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return {'reward_loss':loss}

    def forward(self,z):
        z = tf.expand_dims(tf.convert_to_tensor(z), 0)
        logits = self.reward(z)
        dist = tfpd.Normal(loc=logits,scale=1.0)
        return tf.squeeze(dist.sample()).numpy()
