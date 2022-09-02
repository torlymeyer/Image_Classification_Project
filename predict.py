#IMPORTS
import os
import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import tensorflow_datasets as tfds
from PIL import Image
import argparse

#show TF version
print(tf.__version__)

#BASIC USAGE
#python predict.py test_images/wild_pansy.jpg model_2.h5
#CODE TO RUN FOR TOP 3
#python predict.py test_images/wild_pansy.jpg model_2.h5 --top_k 3
#CODE TO RUN FOR CATEGORY NAMES
#python predict.py test_images/wild_pansy.jpg model_2.h5 --category_names label_map.json


def main(args):
    image_path = args.image_path
    model_path = args.model_path
    top_k = args.top_k
    label_map_path = args.category_names


    print('/n Predicting Image: ', image_path)
    print('/n The model being used is: ', model_path)


    
    model = load_model(model_path)
    probs , classes = predict(image_path, model, top_k)
    
    if label_map_path:

        class_names = load_map(label_map_path)

        outputs = [class_names[str(int(label) + 1)] for label in classes]
    else: 
        outputs = classes

    print(f'The top {top_k} predictions are: {outputs}') #,'/n of these classes', label_map)
        
    
def load_model(model_path):    
    #LOAD MODEL
    
    reloaded_keras_model = tf.keras.models.load_model(model_path, compile = False, custom_objects={'KerasLayer':         hub.KerasLayer})

    # reloaded_keras_model = keras.models.load_model('saved_keras_model',
    # custom_objects={'KerasLayer': hub.KerasLayer}) 
    return reloaded_keras_model

def load_map(map_path):
                    
    #Classes
    with open(map_path, 'r') as f:
        class_names = json.load(f)
    return class_names

#preprocess image func
def process_image(image):
    image = tf.cast(image,tf.float32)
    image = tf.image.resize(image,[224,224])
    image /=255
    return image.numpy()     
                    
#predict func
def predict(image_path, model, top_k):

    image = Image.open(image_path)
    image_array = np.asarray(image)
    processed_img = process_image(image_array)
    img = np.expand_dims(processed_img, axis = 0)
    pred = model.predict(img)
    prob = np.sort(pred)
    top_k_prob = prob[0][-top_k:][::-1].tolist()
    top_k_class = pred.argsort()[0][-top_k:][::-1]
    top_k_class = [str(x) for x in top_k_class]                     
    return   top_k_prob, top_k_class



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Flower Prediction Model')
    parser.add_argument('image_path',
                        help ='path to image')
    parser.add_argument('model_path', 
                        help ='path to model')
    parser.add_argument('--top_k', type = int, default = 5,
                        help = 'top 5 class labels')
    parser.add_argument('--category_names',
                        help = 'labels')
    args = parser.parse_args()
    main(args)

