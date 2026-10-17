import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load your real image
img = tf.keras.utils.load_img(
    "C:\\Users\\yukine\\Downloads\\Machine_Learning_lab\\CNN_lab\\cat.jpg",
)
#image = tf.keras.utils.load_img("url_of_image",   target_size=(128,128))

# Convert image to a numerical array
img_array = tf.keras.utils.img_to_array(img)

# Add batch dimension: (height, width, channels) -> (1, height, width, channels)
img_array = tf.expand_dims(img_array, axis=0)

print("Image shape:", img_array.shape)

plt.imshow(img)
plt.axis("off")
plt.show() 

in the above code what i did is i took an image and converted that image into an array and added a new dimension called batch 
the question arises why we need batch it's coz a tf conv2d expects (batch_size, height, width, channels) and what we had was (height , width , channels(or the number of pixel colours we use))
then why we added axis 0 ? it's coz it tells tf where to add the new dimension which is the 0th place 
