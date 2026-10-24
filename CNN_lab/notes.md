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




model = keras.models.Sequential([
keras.layers.Conv2D(64, 7, activation="relu", padding="same",
input_shape=[28, 28, 1]),
keras.layers.MaxPooling2D(2),
keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
keras.layers.MaxPooling2D(2),
keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
keras.layers.MaxPooling2D(2),
keras.layers.Flatten(),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dropout(0.5),
keras.layers.Dense(64, activation="relu"),
keras.layers.Dropout(0.5),
keras.layers.Dense(10, activation="softmax")
]) 

this is a simple model which feeds on image data and at the end produces 10 outputs which can be used to predict what is the given number or the problem where we predict the numbers which number is written 
we do this in sequence like having a conv layer of 64 filters and 7x7 filters 
then we pool the output down 
later we have again 2 conv layers 
and poool it down again 
and then we have the same conv and pool out again thing 
and we flatten the 3d matrix to a single vector which can be fed to our network 
then we have a dense layer of 128 neurons and we naturally dropout 50% of those to avoid overfitting and we have at last about 10 neurons which outputs about 0 to 9 or the ten digits probability of what we have written 



questions :
. CNN fundamentals
[ ] 
What is a CNN, and why is it used for image processing?


[ ] 
What is a convolutional filter/kernel?


[ ] 
What is a feature map?


[ ] 
What is the difference between convolution and pooling?


[ ] 
Why do CNNs use weight sharing?


[ ] 
What are stride and padding?


[ ] 
What is the difference between valid and same padding?



2. Shapes and calculations
[ ] 
How do you calculate convolution output size?


[ ] 
What does the channel dimension represent?


[ ] 
Why does a CNN output have three dimensions?


[ ] 
How many parameters does Conv2D have?


[ ] 
Why does a 3 × 3 convolution have fewer parameters than a 5 × 5 convolution?



3. Training and activations
[ ] 
Why is ReLU commonly used?


[ ] 
What is backpropagation in a CNN?


[ ] 
What is a loss function?


[ ] 
Why use softmax for classification?


[ ] 
What is overfitting, and how do you reduce it?


[ ] 
What is the difference between training and inference?



4. Architecture and applications
[ ] 
Explain LeNet-5.


[ ] 
Explain AlexNet and why it was important.


[ ] 
Why stack multiple convolution layers?


[ ] 
What is transfer learning?


[ ] 
CNN vs. fully connected neural network?


[ ] 
Where are CNNs used?


Question	What you should be able to explain
Why is a convolution layer's output shape \(H\times W\times C\)?	Spatial dimensions and the number of filters
Why can CNNs require so much GPU memory?	Intermediate activations, gradients, batches, and optimizer states
What happens if stride increases?	Spatial output becomes smaller; computation usually decreases
What happens if you increase the number of filters?	More output channels, parameters, and computation
Why is pooling often used?	Downsampling and some robustness to small spatial shifts
What is a receptive field?	The region of the original input that can influence a particular activation
Why do CNNs need nonlinear activation functions?	Without nonlinearities, stacked layers collapse into an equivalent linear transformation under standard assumptions
Can a CNN work with grayscale images?	Yes; the input channel count is one instead of three
What is batch normalization?	A layer that normalizes activations using batch statistics during training and running statistics at inference
How would you diagnose a model with high training accuracy but low test accuracy?	Overfitting, data leakage checks, augmentation, regularization, and validation strategy
3. Practical coding questions
If you mention that you've implemented CNNs using TensorFlow or Keras, expect questions like these:
1. Build a CNN for MNIST using Keras.
2. Explain Conv2D(32, (3, 3), activation="relu").
3. Why do we use input_shape=(28, 28, 1) for grayscale MNIST?
4. What is the difference between Flatten() and GlobalAveragePooling2D()?
5. Why would you use Dense(10, activation="softmax") for ten-class classification?
6. How do you compile, train, and evaluate a CNN?
7. How do you prevent overfitting when training accuracy rises but validation accuracy stalls?
8. How would you adapt a CNN trained on RGB images to a new dataset with a different number of classes?
Be ready to explain code you actually wrote, not just recite the layer names.
4. Architecture questions: LeNet-5 and AlexNet
Since you've been studying these architectures, know the key distinctions:









LeNet-5
- Designed for handwritten digit recognition.
- Uses convolution and average subsampling.
- Original architecture uses tanh and a special distance-based output.












AlexNet
- Designed for large-scale ImageNet classification.
- Uses ReLU, max pooling, and large fully connected layers.
- Helped demonstrate the value of GPU training and data augmentation.



An interviewer might ask why AlexNet uses convolution layers consecutively, why its first kernel is \(11\times11\), or why dropout is used in its dense layers.