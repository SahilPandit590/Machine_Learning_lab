import tensorflow as tf 
import numpy as np 
import pandas as pd 

image = np.arrange(1,36,dtype = np.float32).reshape(1,5,7,1)
conv = tf.keras.layers.Conv2D(
    filters=1,
    kernel_size=(3,3),
    strides=(2,2),
    padding='same',
    use_bias=False,
    kernel_initializer="ones"
)
print(image)
output = conv(image)
print("Input shape:", image.shape)
print("Output shape:", output.shape)
print("Feature map:\n", output.numpy()[0, :, :, 0])