import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import DirectoryIterator, load_img

from typing import Tuple
from PIL.JpegImagePlugin import JpegImageFile

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
    
    def fit(self, data: DirectoryIterator, epoch: int) -> None:
        '''
        Train model on the given data
        '''

        self.class_mapping = {data.class_indices[c]:c for c in data.class_indices}
        self.model.fit(data, epochs=epoch)

    def evaluate(self, data: DirectoryIterator) -> np.float64:
        '''
        Evaluate model accuracy on the given data
        '''

        return self.model.evaluate(data)

    def predict(self, image: np.ndarray) -> Tuple[int, str]:
        '''
        Make prediction on the given image and return predicted class index and name

        :param image: np.ndarray object representing image to predict
        :returns: A tuple containing prediction class index and name
        '''

        assert image.ndim == 3 and image.shape[0] == image.shape[1] == 150 and image.shape[2] == 3, 'An image must be in shape (150,150,3)'

        pred_class = self.model.predict_classes(image.reshape(1,150,150,3))[0]
        return (pred_class, self.class_mapping[pred_class])