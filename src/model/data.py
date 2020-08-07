import tensorflow.keras.preprocessing as preprocessing
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import DirectoryIterator

DATA_PATH = '../data/'
loader = preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

def load_train_data() -> DirectoryIterator:
    '''
    Load train data and infer classes from the directory structure
    '''

    return loader.flow_from_directory(directory=f'{DATA_PATH}train/', target_size=(150,150))

def load_test_data() -> DirectoryIterator:
    '''
    Load test data and infer classes from the directory structure
    '''

    return loader.flow_from_directory(directory=f'{DATA_PATH}test/', target_size=(150,150))

def load_predict_data():
    '''
    Load predict data (To be implemented)
    '''

    pass