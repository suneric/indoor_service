import os, sys
from tensorflow import keras
from .network import *

class CycleGAN(keras.Model):
    def __init__(self,image_shape, lambda_c=10.0, lambda_i=0.5):
        super().__init__()
        self.gen_G = Generator(image_shape,name="generator_G")
        self.gen_F = Generator(image_shape,name="generator_F")
        self.disc_X = Discriminator(image_shape,name="discriminator_X")
        self.disc_Y = Discriminator(image_shape,name="discriminator_Y")
        self.lambda_cycle = lambda_c
        self.lambda_identity = lambda_i

    def compile(
        self,
        gen_G_opt,
        gen_F_opt,
        disc_X_opt,
        disc_Y_opt,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super().compile()
        self.gen_G_optimizer = gen_G_opt
        self.gen_F_optimizer = gen_F_opt
        self.disc_X_optimizer = disc_X_opt
        self.disc_Y_optimizer = disc_Y_opt
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self,batch_data):
        real_x, real_y = batch_data
        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.gen_G(real_x,training=True)
            fake_x = self.gen_F(real_y,training=True)
            cycled_x = self.gen_F(fake_y,training=True)
            cycled_y = self.gen_G(fake_x,training=True)
            same_x = self.gen_F(real_x,training=True)
            same_y = self.gen_G(real_y,training=True)
            disc_real_x = self.disc_X(real_x,training=True)
            disc_fake_x = self.disc_X(fake_x,training=True)
            disc_real_y = self.disc_Y(real_y,training=True)
            disc_fake_y = self.disc_Y(fake_y,training=True)
            gen_loss_G = self.generator_loss_fn(disc_fake_y)
            gen_loss_F = self.generator_loss_fn(disc_fake_x)
            cycle_loss_G = self.cycle_loss_fn(real_y,cycled_y)*self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x,cycled_x)*self.lambda_cycle
            id_loss_G = (self.identity_loss_fn(real_y,same_y)*self.lambda_cycle*self.lambda_identity)
            id_loss_F = (self.identity_loss_fn(real_x,same_x)*self.lambda_cycle*self.lambda_identity)
            total_loss_G = gen_loss_G + cycle_loss_G + id_loss_G
            total_loss_F = gen_loss_F + cycle_loss_F + id_loss_F
            disc_loss_X = self.discriminator_loss_fn(disc_real_x,disc_fake_x)
            disc_loss_Y = self.discriminator_loss_fn(disc_real_y,disc_fake_y)
        grads_G = tape.gradient(total_loss_G,self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F,self.gen_F.trainable_variables)
        grads_X = tape.gradient(disc_loss_X,self.disc_X.trainable_variables)
        grads_Y = tape.gradient(disc_loss_Y,self.disc_Y.trainable_variables)
        self.gen_G_optimizer.apply_gradients(zip(grads_G,self.gen_G.trainable_variables))
        self.gen_F_optimizer.apply_gradients(zip(grads_F,self.gen_F.trainable_variables))
        self.disc_X_optimizer.apply_gradients(zip(grads_X,self.disc_X.trainable_variables))
        self.disc_Y_optimizer.apply_gradients(zip(grads_Y,self.disc_Y.trainable_variables))
        return {
            "G_loss":total_loss_G,
            "F_loss":total_loss_F,
            "X_loss":disc_loss_X,
            "Y_loss":disc_loss_Y,
        }

    def save(self, model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        self.gen_G.save_weights(os.path.join(model_path,"gen_G"))
        self.gen_F.save_weights(os.path.join(model_path,"gen_F"))
        self.disc_X.save_weights(os.path.join(model_path,"disc_X"))
        self.disc_Y.save_weights(os.path.join(model_path,"disc_Y"))

    def load(self, model_path):
        self.gen_G.load_weights(os.path.join(model_path,"gen_G"))
        self.gen_F.load_weights(os.path.join(model_path,"gen_F"))
        self.disc_X.load_weights(os.path.join(model_path,"disc_X"))
        self.disc_Y.load_weights(os.path.join(model_path,"disc_Y"))
