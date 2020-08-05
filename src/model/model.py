import tensorflow as tf
from tensorflow import keras
from keras.applications.xception import Xception, preprocess_input
from keras.layers import Flatten, Dense
from keras import Model

from data import load_train_data, load_test_data

# Load data
train_data = load_train_data()

# Transfer Xception model
xception_input = keras.Input(shape=train_data.image_shape)
xception = Xception(include_top=False, input_tensor=xception_input)

# Classifier layers
flat = Flatten()(xception.outputs)
hidden1 = Dense(1536, activation='relu')(flat)
hidden2 = Dense(512, activation='relu')(hidden1)
output = Dense(train_data.num_classes, activation='softmax')(hidden2)

# DpicNet
model = Model(inputs=xception.inputs, outputs=output, name='DpicNet')

print(model.summary())