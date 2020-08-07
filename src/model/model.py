import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import DirectoryIterator

from typing import Tuple

class DpicNet():
    def __init__(self, input_shape: Tuple[int, int, int], classes: int, hidden1_nodes: int, hidden2_nodes: int, learning_rate: float):
        # DpicNet model
        self.model = Sequential(name='DpicNet')

        # Transfer Xception model
        xception = Xception(include_top=False, input_shape=input_shape)

        ## Mark as nontrainable
        for layer in xception.layers:
            layer.trainable = False
        
        ## Add to DpicNet
        self.model.add(xception)

        ## Flatten before classifier layers    
        self.model.add(Flatten())

        ## Classifier layers
        self.model.add(Dense(hidden1_nodes, activation='relu'))
        self.model.add(Dense(hidden2_nodes, activation='relu'))
        self.model.add(Dense(classes, activation='softmax'))

        # DpicNet
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def __str__(self):
        self.model.summary()
        return ''
    
    def fit(self, data: DirectoryIterator, epoch: int):
        self.model.fit(data, epochs=epoch)

    def evaluate(self, data: DirectoryIterator):
        return self.model.evaluate(data)

    def predict(self, images):
        pass