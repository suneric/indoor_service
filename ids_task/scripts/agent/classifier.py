import os, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .model import *
from .util import *

class LatentClassifier(keras.Model):
    def __init__(self,z_dim,output_unit=10,lr=1e-3):
        super().__init__()
        self.angle = latent_reward(z_dim,output_unit=output_unit)
        self.optimizer = keras.optimizers.Adam(lr)
        self.loss_fn = keras.losses.CategoricalCrossentropy()
        self.output_unit = output_unit

    def train(self,buffer,encoder,epochs=100,batch_size=32):
        print("training latent reward classifier, epochs {}, batch_size {}".format(epochs,batch_size))
        for _ in range(epochs):
            idxs = np.random.choice(buffer.size,batch_size)
            data = buffer.get_observation(idxs)
            image = tf.convert_to_tensor(data['image'])
            angle = tf.one_hot(angle_class_index(data['angle']),depth=self.output_unit)
            mu,logv,z = encoder(image)
            with tf.GradientTape() as tape:
                angle_pred = self.angle(z)
                loss = self.loss_fn(angle,angle_pred)
            grad = tape.gradient(loss,self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad,self.trainable_variables))
            print("reward classifier loss {}".format(loss))

    def save(self, model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        self.angle.save_weights(model_path)

    def load(self, model_path):
        self.angle.load_weights(model_path)
