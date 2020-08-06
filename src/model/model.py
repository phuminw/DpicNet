import tensorflow as tf
from tensorflow import keras
from keras.applications.xception import Xception, preprocess_input
from keras.layers import Flatten, Dense
from keras import Model

# from data import load_train_data, load_test_data

class DpicNet():
    def __init__(self, hidden1_nodes: int, hidden2_nodes: int):
        # # Load data
        # self.train_data = load_train_data()

        # Transfer Xception model
        self.xception_input = keras.Input(shape=train_data.image_shape)
        self.xception = Xception(include_top=False, input_tensor=self.xception_input)

        ## Mark as nontrainable
        for layer in self.xception.layers:
            layer.trainable = False

        ## Flatten before classifier layers    
        self.flat = Flatten()(self.xception.outputs)

        # Classifier layers
        self.hidden1 = Dense(hidden1_nodes, activation='relu')(self.flat)
        self.hidden2 = Dense(hidden2_nodes, activation='relu')(self.hidden1)
        self.output = Dense(train_data.num_classes, activation='softmax')(self.hidden2)

        # DpicNet
        self.model = Model(inputs=self.xception.inputs, outputs=self.output, name='DpicNet')

    def __str__(self):
        return model.summary()
    
    def fit(data: tf.Tensor):
        pass

    def evaluate(data: tf.Tensor):
        pass

    def predict(images):
        pass