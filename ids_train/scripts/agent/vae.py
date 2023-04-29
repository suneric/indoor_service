"""
reference
https://keras.io/examples/generative/vae/
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from agent.model import fv_decoder, fv_encoder

class FVVAE(keras.Model):
    def __init__(self, image_shape, force_dim, latent_dim, lr=1e-4,**kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = fv_encoder(image_shape, force_dim, latent_dim)
        self.decoder = fv_decoder(latent_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    def train_step(self,images,forces):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([images,forces])
            r_images, r_forces = self.decoder(z)
            # reconstruction image and force
            image_loss = tf.reduce_sum(keras.losses.MSE(images,r_images), axis=(1,2))
            force_loss = keras.losses.MSE(forces,r_forces)
            rc_loss = tf.reduce_mean(image_loss) + tf.reduce_mean(force_loss)
            # augmented kl loss per dim
            kl_loss = -0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = rc_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": rc_loss,
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
