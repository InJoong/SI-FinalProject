import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.core import Dropout

df_train = pd.read_csv('./dataset/sign_mnist_train/sign_mnist_train.csv')
df_test = pd.read_csv('./dataset/sign_mnist_test/sign_mnist_test.csv')

# Separating label and data 
y_train = df_train['label']
X_train = df_train.drop(['label'], axis=1)

y_test = df_test['label']
X_test = df_test.drop(['label'], axis=1)

# Change shape of the images
size = 28
channel = 1
X_train = X_train.values.reshape(df_train.shape[0], size, size, channel)
X_test = X_test.values.reshape(df_test.shape[0], size, size, channel)

# Clamp int color value to float color value
X_train = X_train / 255
X_test = X_test / 255

# Determine the architecture of my NN
model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=[28, 28]),   # Add the input layer
  keras.layers.Dense(500, activation="relu"),   # Add the hidden layer
  keras.layers.Dropout(0.2),
  keras.layers.Dense(300, activation="relu"),   # Add the hidden layer
  keras.layers.Dropout(0.2),
  keras.layers.Dense(25, activation="softmax")  # Add the output layer
])

# Set the parameters for our model
model.compile(
  loss="sparse_categorical_crossentropy",
  optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.01), # Reduce learning rate in case the model is changing to fast
  metrics=["accuracy"]
)

# Start training the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Display the results
# Model values
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.subplot(2,2,2)
plt.plot(range(len(history.history['loss'])), history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epochs')
# Validation values
plt.subplot(2,2,3)
plt.plot(range(len(history.history['val_accuracy'])), history.history['val_accuracy'])
plt.ylabel('val_accuracy')
plt.xlabel('epochs')
plt.subplot(2,2,4)
plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'])
plt.ylabel('val_loss')
plt.xlabel('epochs')
plt.show()
