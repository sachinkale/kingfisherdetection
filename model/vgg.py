

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os

from PIL import Image
import re 
import json


from keras.models import load_model
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 

from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D

from keras.applications.vgg16 import VGG16,preprocess_input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


batch_size = 32
initial_epochs = 20
IMG_HEIGHT = 224
IMG_WIDTH = 224

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=0.2,
                                   rotation_range=45,
                                   width_shift_range=.15,
                                   height_shift_range=.15,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('train3_cropped_resized',target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=batch_size,class_mode='categorical')

#training_set.image_shape


# prepare the image for the VGG model

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_set = test_datagen.flow_from_directory('test3_cropped_resized',
                                             target_size = (IMG_HEIGHT, IMG_WIDTH),
                                             batch_size = batch_size,
                                             class_mode = 'categorical')



#get total count and label of classes 
class_names = glob("./test3_cropped_resized/*") # Reads all the folders in which images are present
for i in class_names:
    m = re.search("/([^/]+Kingfisher)", i,re.IGNORECASE)
    if m:
        class_names[class_names.index(i)] = m.group(1)
        
        
class_names = sorted(class_names) # Sorting them
num_of_classes = len(class_names)
name_id_map = dict(zip(range(len(class_names)),class_names ))


base_model = VGG16(weights='imagenet', include_top=False)
for layers in base_model.layers:
   layers.trainable = False
    
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_of_classes, activation='softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
base_learning_rate = 0.0001
model.compile(optimizer=Adam(lr=base_learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()


history = model.fit(
    x = training_set,
    epochs=initial_epochs,
    validation_data=test_set
    
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

for layers in model.layers[:19]:
   layers.trainable = True
for layers in model.layers[19:]:
    layers.trainable = False
    
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=Adam(lr=base_learning_rate/10), loss='categorical_crossentropy', metrics = ['accuracy'])

fine_tune_epochs = 10   
total_epochs =  initial_epochs + fine_tune_epochs


filepath="vggK5-weights-best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history_fine = model.fit(training_set,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=test_set,
                         callbacks=callbacks_list)


acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()



#save classes associated with model for future use
j = json.dumps(name_id_map, indent=4)
f = open('vggKFclassIdMap.json', 'w')
print(j, file=f)
f.close()
