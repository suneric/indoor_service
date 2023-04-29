"""
reference
https://keras.io/examples/generative/vae/
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras

class Sampling(keras.layers.Layer):
    """Use (mean,log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        mean, logv = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch,dim))
        return mean + tf.exp(0.5*logv)*eps

def conv_encoder(image_shape, force_dim, latent_dim):
    v_input = keras.Input(shape=image_shape)
    vh = keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(v_input)
    vh = keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=2,padding='same',activation='relu')(vh)
    vh = keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=2,padding='same',activation='relu')(vh)
    vh = keras.layers.Flatten()(vh)
    vh = keras.layers.Dense(32,activation='relu')(vh)

    f_input = keras.Input(shape=(force_dim))
    fh = keras.layers.Dense(32, activation='relu')(f_input)
    fh = keras.layers.Dense(16, activation='relu')(fh)

    h = keras.layers.concatenate([vh, fh])
    h = keras.layers.Dense(32,activation='relu')(h)
    z_mean = keras.layers.Dense(latent_dim, name='z_mean')(h)
    z_logv = keras.layers.Dense(latent_dim,name='z_logv')(h)
    z = Sampling()([z_mean,z_logv])
    model = keras.Model([v_input,f_input],[z_mean,z_logv,z],name='encoder')
    print(model.summary())
    return model

def conv_decoder(latent_dim):
    input = keras.Input(shape=(latent_dim,))
    h = keras.layers.Dense(32,activation='relu')(input)
    h = keras.layers.Dense(8*8*32+16, activation='relu')(h)
    vh = keras.layers.Lambda(lambda x: x[:,0:8*8*32])(h)
    vh = keras.layers.Reshape((8,8,32))(vh)
    vh = keras.layers.Conv2DTranspose(filters=8,kernel_size=3,strides=2,padding='same',activation='relu')(vh)
    vh = keras.layers.Conv2DTranspose(filters=16,kernel_size=3,strides=2,padding='same',activation='relu')(vh)
    vh = keras.layers.Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same',activation='relu')(vh)
    v_output = keras.layers.Conv2DTranspose(filters=1,kernel_size=3,padding='same',activation='sigmoid')(vh)

    fh = keras.layers.Lambda(lambda x: x[:,8*8*32:])(h)
    fh = keras.layers.Dense(32,activation='relu')(fh)
    f_output = keras.layers.Dense(3, activation='linear')(fh)

    model = keras.Model(input,[v_output,f_output],name='decoder')
    print(model.summary())
    return model


class ConvVAE(keras.Model):
    def __init__(self, image_shape, force_dim, latent_dim, lr=1e-4,**kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = conv_encoder(image_shape, force_dim, latent_dim)
        self.decoder = conv_decoder(latent_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    def train_step(self,images,forces):
        with tf.GradientTape() as tape:
            mean, logv, z = self.encoder([images,forces])
            y_images, y_forces = self.decoder(z)
            image_loss = tf.reduce_sum(keras.losses.binary_crossentropy(images,y_images), axis=(1,2))
            force_loss = keras.losses.MSE(forces,y_forces)
            reconstruction_loss = tf.reduce_mean(image_loss) + tf.reduce_mean(force_loss)
            kl_loss = tf.reduce_sum(-0.5*(1+logv-tf.square(mean)-tf.exp(logv)), axis=1)
            kl_loss = tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def learn(self, data, size, epochs=100, batch_size=64):
        print("training epoches {}, batch size {}/{}".format(epochs,batch_size,size))
        (image_buf,force_buf,action_buf,return_buf,advantage_buf,logprob_buf) = data
        #self.fit(image_buf,epochs=epochs,batch_size=batch_size)
        for epoch in range(epochs):
            idxs = np.random.choice(size,batch_size)
            images = tf.convert_to_tensor(image_buf[idxs])
            forces = tf.convert_to_tensor(force_buf[idxs])
            info = self.train_step(images,forces)
            print("epoch {}, loss {:.4f}, reconstruction loss {:.4f}, kl loss {:.4f}".format(
                    epoch,
                    info["loss"],
                    info["reconstruction_loss"],
                    info["kl_loss"]
                    )
                )

    def save(self, encoder_path, decoder_path):
        if not os.path.exists(os.path.dirname(encoder_path)):
            os.makedirs(os.path.dirname(encoder_path))
        self.encoder.save_weights(encoder_path)
        if not os.path.exists(os.path.dirname(decoder_path)):
            os.makedirs(os.path.dirname(decoder_path))
        self.decoder.save_weights(decoder_path)

    def load(self, encoder_path, decoder_path):
        self.encoder.load_weights(encoder_path)
        self.decoder.load_weights(decoder_path)
