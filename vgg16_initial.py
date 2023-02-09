import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load train and test csv files into pandas dataframes
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Pre-process the images and resize them
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    return img

# Prepare the train and test data
train_data = []
for i, row in train_df.iterrows():
    img_path = row["ImagePath"]
    img = preprocess_image(img_path)
    train_data.append(img)

train_data = np.array(train_data)
train_labels = train_df["ID"].values

test_data = []
for i, row in test_df.iterrows():
    img_path = row["ImagePath"]
    img = preprocess_image(img_path)
    test_data.append(img)

test_data = np.array(test_data)
test_labels = test_df["ID"].values

# Split the train data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Load the VGG16 model with pre-trained weights
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze the layers
for layer in vgg_model.layers:
    layer.trainable = False

# Add the fully connected layers
model = Sequential()
model.add(vgg_model)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(31, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))

model.save("hand_identification_model.h5")

# Evaluate the model on test data
# test_loss, test_acc = model.evaluate(test_data, test_labels)
# print('Test Accuracy: ', test_acc)
# print('Test loss:', test_loss)
