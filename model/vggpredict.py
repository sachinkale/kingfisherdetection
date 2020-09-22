from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import json


model = load_model('vggK5-weights-best.h5')

img_path ='_DSC3048n.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
f = open('vggKFclassIdMap.json')
name_id_map = json.load(f)


score = tf.nn.softmax(preds[0])

result = str(np.argmax(score))

print("This image most likely belongs to \x1b[6;30;42m{}\x1b[0m".format(name_id_map[result]))
