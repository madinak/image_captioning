import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np


# prepare images
def load_image_captions(image_path, captions, size=(224, 224)):  
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size)
    image = preprocess_input(image) 
    return image, captions