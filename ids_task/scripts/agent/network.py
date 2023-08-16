import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

"""
Implements Reflection Padding as a Layer
"""
class ReflectionPadding2D(layers.Layer):
    def __init__(self,padding=(1,1),**kwargs):
        self.padding = tuple(padding)
        super().__init__(**kwargs)

    def call(self,input_tensor,mask=None):
        padding_width,padding_height = self.padding
        padding_tensor = [
            [0,0],
            [padding_height,padding_height],
            [padding_width,padding_width],
            [0,0],
        ]
        return tf.pad(input_tensor,padding_tensor,mode="REFLECT")

"""
A Residual Neural Network, a.k.a ResNet is a model in which the weight layers
learn residual functions with reference to the layer inputs.
A ResNet is a network with skip connection that perform identity mappings,
merged with the layer outputs by addition. This enables deep learning models\
with tens or hundreds of layers to train easily and approach better accuracy when
going deeper. The identity skip connections, often referred to as "residual connections".
"""
def residual_block(
    x,
    activation,
    kernel_size=(3,3),
    strides=(1,1),
    padding="valid",
    use_bias=False
):
    kernel_init = keras.initializers.RandomNormal(mean=0.0,stddev=0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0,stddev=0.02)
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer = kernel_init,
        padding = padding,
        use_bias = use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer = kernel_init,
        padding = padding,
        use_bias = use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = layers.add([input_tensor,x])
    return x

"""
Reduce 2D dimensions, the width and height, of the image by the stride.
The stride is the length of the step the filter takes.
"""
def downsample(
    x,
    filters,
    activation,
    kernel_size=(3,3),
    strides=(2,2),
    padding="same",
    use_bias=False
):
    kernel_init = keras.initializers.RandomNormal(mean=0.0,stddev=0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0,stddev=0.02)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides = strides,
        kernel_initializer = kernel_init,
        padding = padding,
        use_bias = use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    if activation:
        x = activation(x)
    return x

"""
Increase the dimension of the image, Conv2DTranspose does basically the opposite
of a Conv2D layer.
"""
def upsample(
    x,
    filters,
    activation,
    kernel_size=(3,3),
    strides=(2,2),
    padding = "same",
    use_bias=False,
):
    kernel_init = keras.initializers.RandomNormal(mean=0.0,stddev=0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0,stddev=0.02)
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides = strides,
        padding = padding,
        kernel_initializer=kernel_init,
        use_bias = use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    if activation:
        x = activation(x)
    return x

def Generator(
    image_shape,
    filters=64,
    num_downsampling=2,
    num_residual=9,
    num_upsampling=2,
    name=None
):
    kernel_init = keras.initializers.RandomNormal(mean=0.0,stddev=0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0,stddev=0.02)
    img_input = layers.Input(shape=image_shape)
    x = ReflectionPadding2D(padding=(3,3))(img_input)
    x = layers.Conv2D(filters,(7,7),kernel_initializer=kernel_init,use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = layers.Activation('relu')(x)
    for _ in range(num_downsampling):
        filters *= 2
        x = downsample(x,filters=filters,activation=layers.Activation('relu'))
    for _ in range(num_residual):
        x = residual_block(x,activation=layers.Activation('relu'))
    for _ in range(num_upsampling):
        filters //= 2
        x = upsample(x,filters,activation=layers.Activation('relu'))

    x = ReflectionPadding2D(padding=(3,3))(x)
    x = layers.Conv2D(1,(7,7),padding='valid')(x)
    x = layers.Activation('tanh')(x)
    model = keras.models.Model(img_input,x,name=name)
    return model

def Discriminator(
    image_shape,
    filters=64,
    num_downsampling=3,
    name=None
):
    kernel_init = keras.initializers.RandomNormal(mean=0.0,stddev=0.02)
    img_input = layers.Input(shape=image_shape)
    x = layers.Conv2D(filters,(4,4),strides=(2,2),padding='same',kernel_initializer=kernel_init)(img_input)
    x = layers.LeakyReLU(0.2)(x)
    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4,4),
                strides=(2,2),
            )
        else:
            x = downsample(
                x,filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4,4),
                strides=(1,1),
            )
    x = layers.Conv2D(1,(4,4),strides=(1,1),padding='same',kernel_initializer=kernel_init)(x)
    model = keras.models.Model(inputs=img_input,outputs=x,name=name)
    return model
