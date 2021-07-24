
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Arguments parser
parser = argparse.ArgumentParser (description ='Image Classifier')
parser.add_argument ('-i','--image_path', default='./test_images/cautleya_spicata.jpg', action = 'store', help = 'Your Image Path', type = str)
parser.add_argument ('-m','--model', default='./best_model.h5', action = 'store', help='Your Model Path', type = str)
parser.add_argument ('-t','--top_k', default = 5,  dest = 'top_k', action = 'store',help = 'Return Top K Probabilities', type = int)
parser.add_argument ('-c','--category', dest = 'category_names', action = 'store', default ='label_map.json' , help = 'Maps leable with names')

arg_parser = parser.parse_args()
image_path = arg_parser.image_path
model_path = arg_parser.model
topk = arg_parser.top_k
category_names = arg_parser.category_names

# TODO: Create the process_image function
def process_image(image):
    image_size = 224
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (image_size, image_size))
    image /=255
    image = image.numpy()
    return image

# TODO: Create the predict function
def predict(image_path,  model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis = 0)
    prediction = model.predict(processed_image)
    
    probs, classes = tf.math.top_k(prediction,top_k)
    
    probs = probs.numpy().squeeze()
    classes = classes.numpy().squeeze()
    classes = [str(i) for i in classes]
    return probs, classes

if __name__ == "__main__":
    print('Start predection ......')
          
          # loads json T map leables 
    with open('label_map.json', 'r') as f:
          class_names = json.load(f)
          # Model loading
          model = tf.keras.models.load_model('./best_model.h5', custom_objects ={'KerasLayer':hub.KerasLayer})
          
          probs, classes = predict(image_path, model, topk)
            
          label_names = [class_names[str(int(idd)+1)] for idd in classes]          
          print('probs: ', probs)
          print('Classes: ', classes)
          print('label Names: ', label_names)
            
          print('Done .....')



