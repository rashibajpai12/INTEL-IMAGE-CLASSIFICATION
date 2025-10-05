# INTEL-IMAGE-CLASSIFICATION

In this project, we will be working on Intel images i.e. images of buildings, forest, street, etc. We will built a convolutional neural network and train it on this images. This is a multi class classification problem and we will use Keras.

# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')

First we will mount our google drive on colab so that we can use the dataset directly from our drive. For this you first need to upload the data on your drive and then mount the drive on colab.
# After executing the cell above, Drive files will be present in "/content/drive/My Drive".
!ls "/content/drive/My Drive"

After mounting our drive we will locate the folder where our data is stored to use it in our colab notebook. Here we will see all the folders I have in my drive and 'Intel Image Dataset' contains the images that we will work on.

# Important imports
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.image import imread

import cv2

import random

from os import listdir

from sklearn.preprocessing import  LabelBinarizer

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array, array_to_img

from keras.optimizers import Adam

from PIL import Image

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LeakyReLU

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from matplotlib.image import imread

import random

from os import listdir

# ✅ Plotting 25 random images from the 'mountain' class

plt.figure(figsize=(11,11))

# Correct path based on your Drive structure

path = "/content/drive/My Drive/Intel_Image_Dataset/mountain"

for i in range(1, 26):

    plt.subplot(5, 5, i)
    
    plt.tight_layout()

    # Select a random image from the folder
    
    rand_img = imread(path + '/' + random.choice(sorted(listdir(path))))

    plt.imshow(rand_img)
    
    plt.title('mountain')
    
    plt.xlabel(str(rand_img.shape[1]), fontsize=10)
    
    plt.ylabel(str(rand_img.shape[0]), fontsize=10)

plt.show()

# Let's visualize some of the mountain images that we will be working on. Also we will observe x and y dimensions of the image.

# Setting root directory path and creating empty list

dir = "/content/drive/My Drive/Intel_Image_Dataset/mountain"

root_dir = listdir(dir)

image_list, label_list = [], []

Setting the root directory for the dataset and storing all the folders name of the dataset. We will also create 2 empty list for image and lables.

# Reading and converting image to numpy array
from os import listdir
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Assuming 'dir' is your main dataset directory
dir = "/content/drive/My Drive/Intel_Image_Dataset"
root_dir = listdir(dir)

image_list = []
label_list = []

# Reading and converting image to numpy array
for directory in root_dir:
    folder_path = f"{dir}/{directory}"

    # ✅ Skip if it's not a folder (prevents NotADirectoryError)
    if not os.path.isdir(folder_path):
        continue

    for files in listdir(folder_path):
        image_path = f"{folder_path}/{files}"

        # Skip if it's not an image (extra safety)
        if not files.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image = Image.open(image_path)
        image = image.resize((150,150))  # All images resized to same dimension
        image = img_to_array(image)
        image_list.append(image)
        label_list.append(directory)

Next we need to resize images as some of the images don't have same dimensions. So, we will read and resize all the images. Then we will convert it into array and appending the list created above with the image and its label.


# Visualize the number of classes count
label_counts = pd.DataFrame(label_list).value_counts()
label_counts

Checking for images per class.

# Checking count of classes
num_classes = len(label_counts)
num_classes

Storing the number of classes which will be used further in model architecture.

# Checking x data shape
np.array(image_list).shape

Check the shape of the x data for input layer of model architecture.

# Checking y data shape
label_list = np.array(label_list)
label_list.shape

Checking the number of labels in y data which should be equal to total number of images.

# Splitting dataset into test and train
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state = 10) 

Now we will split our dataset into testing and training using train_test_split() from sklearn.

# Normalize and reshape data
x_train = np.array(x_train, dtype=np.float16) / 225.0
x_test = np.array(x_test, dtype=np.float16) / 225.0
x_train = x_train.reshape( -1, 150,150,3)
x_test = x_test.reshape( -1, 150,150,3)

Next we will normalize the images by dividing them with 255 and we will also reshape x_train and x_test data. 

# Binarizing labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
print(lb.classes_)

Here we will use label binarizer to one hot encode our y data. We will also print the sequence of the classes.

# Splitting the training data set into training and validation data sets

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

Now we will split the training data to validation and training data.

# Creating model architecture
model = Sequential([
        Conv2D(16, kernel_size = (3,3), input_shape = (150,150,3)),
        BatchNormalization(),
        LeakyReLU(),
          
        Conv2D(32, kernel_size = (3,3)),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling2D(5,5),
        
        Conv2D(64, kernel_size = (3,3)),
        BatchNormalization(),
        LeakyReLU(),
        
        Conv2D(128, kernel_size = (3,3)),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling2D(5,5),

        Flatten(),
    
        Dense(64),
        Dropout(rate = 0.2),
        BatchNormalization(),
        LeakyReLU(),
        
        Dense(32),
        Dropout(rate = 0.2),
        BatchNormalization(),
        LeakyReLU(),
    
        Dense(16),
        Dropout(rate = 0.2),
        BatchNormalization(),
        LeakyReLU(1),
    
        Dense(6, activation = 'softmax')    
        ])
model.summary()

Now we will create a network architecture for the model. We have used different types of layers according to their features namely BatchNormalization (Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch), LeakyRelu (The Leaky ReLU modifies the function to allow small negative values when the input is less than zero), Conv_2d (It is used to create a convolutional kernel that is convolved with the input layer to produce the output tensor), max_pooling2d (It is a downsampling technique which takes out the maximum value over the window defined by poolsize), flatten (It flattens the input and creates a 1D output), Dense (Dense layer produce the output as the dot product of input and kernel). In the last layer we will use softmax as the activation function because it is a multi class classification problem.

# Compiling model
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0005),metrics=['accuracy'])

For compiling the model we need to pass 3 parameters namely loss, optimizer and metrics. Here we will use loss as categorical_crossentropy, optimizer as Adam and metrics as accuracy.

# Training the model
epochs = 70
batch_size = 128
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_val, y_val))

Fitting the model with the data and finding out the accuracy at each epoch to see how our model is learning. Now we will train our model on 70 epochs and a batch size of 128. You can try using more number of epochs to increase accuracy. During each epochs we can see how the model is performing by viewing the training and validation accuracy.

# Saving model
model.save("/content/drive/My Drive/intel_image.h5")

We will save the model using model.save() to use it later for prediction.

#Plot the training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
plt.show()

Next we will plot the accuracy of the model for the training history.


#Plot the loss history
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], color='r')
plt.plot(history.history['val_loss'], color='b')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
plt.show()

Next we will plot the loss of the model for the training history.


# Calculating test accuracy 
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

Evaluating the model to know the accuracy of the model on the test data.

# Storing model predictions
y_pred = model.predict(x_test)

Generating predictions for test data and storing them into y_pred.

# Plotting image to compare
img = array_to_img(x_test[1])
img

Visualizing an image to be predicted in further steps.

# Finding max value from predition list and comaparing original value vs predicted
labels = lb.classes_
print(labels)
print("Originally : ",labels[np.argmax(y_test[1])])
print("Predicted : ",labels[np.argmax(y_pred[1])])

Now, we will create list of labels using object of label binarizer. We will print that list and finally we will print out the prediction and the original label of the image we visualized above using argmax().

## Conclusion:

In this project we saw how we can create a CNN using different layers. Normalizing is an important step when working with any type of dataset. We will use this model to predict the class of the image supplied to the model.

## Scope:

This project has a vast scope, it can be used to classify satellite images, drone images, google images into different classes like sea, mountain, etc. 




