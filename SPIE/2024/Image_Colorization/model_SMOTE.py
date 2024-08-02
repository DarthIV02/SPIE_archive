import itertools
import os

import matplotlib.pylab as plt
import numpy as np
import csv
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

print("TF version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

#SMOTE
import pandas as pd
import numpy as np
import os
import sys
from shutil import copyfile
import os.path
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array,load_img
from PIL import Image
from sklearn.model_selection import train_test_split
from numpy import load
import matplotlib.pyplot as plt

seed = 150 # 20 works
tf.random.set_seed(seed) # 1 tf 40 np = 32 tops
np.random.seed(seed)

import keras
print(keras.__version__)

imagegen = ImageDataGenerator()

# load train data from drive
images = []
target = []
paths = []
with open('./dataset_64x64_cpsnr_3mod.csv', newline='') as csvfile:
    dataset = csv.reader(csvfile, delimiter=',')
    alpha = .3
    smooth = lambda a, y_hot: (1 - a) * y_hot + a / 4
    for i, row in enumerate(dataset):
        if i > 0:
            if row[0] == 'landscape':

                # image
                image_path = f'/mnt/data/dataset/landscape_dat/landscape_64x64_bw/{row[1]}'

            elif row[0] == 'imagenet':

                # image
                image_path = f'/mnt/data/dataset/imagenet/imagenet_64x64_bw/{row[1]}'
            else:
                print("ERROR")

            paths.append(image_path)

            image = Image.open(image_path)
            data = np.asarray(image)
            images.append(data)

            #classification target
            if row[2]=='siggraph_2016':
                #target.append(np.array([1,0,0,0]))
                #target.append(smooth(alpha, np.array([1,0,0,0])))
                target.append(0)
            elif row[2]=='zhang_2016':
                #target.append(np.array([0,0,1,0]))
                #target.append(smooth(alpha, np.array([0,0,1,0])))
                target.append(1)
            elif row[2]=='ddcolor':
                #target.append(np.array([0,0,0,1]))
                #target.append(smooth(alpha, np.array([0,0,0,1])))
                target.append(2)
            else:
                print(row)
                x = input("Enter")

images = np.array(images)
rgb_batch = np.repeat(images[..., np.newaxis], 3, -1)
target = np.array(target)
paths = np.array(paths)

print(images.shape)
rgb_batch = rgb_batch.reshape(22698,64*64*3)
print(target.shape)

X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(rgb_batch, target, paths, test_size=0.2, random_state=1)

#Apply SMOTE method 
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_smote, y_smote = sm.fit_resample(X_train, y_train)

print("Done SMOTE")

Xsmote_img=X_smote.reshape(23574,64,64,3)

print(Xsmote_img.shape)
print(y_smote.shape)

print(X_train.shape)
print(X_test.shape)
print(paths_train.shape)
print(paths_test.shape)

X_train, X_val, y_train, y_val = train_test_split(Xsmote_img, y_smote, test_size=0.125, random_state=1)

X_train, X_val, X_test = X_train / 255., X_val / 255., X_test / 255.
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

y_train_exp = []
for i in range(y_train.shape[0]):
    temp = np.zeros(3)
    temp[y_train[i]] = 1
    y_train_exp.append(temp)
y_train_exp = np.array(y_train_exp)

y_test_exp = []
for i in range(y_test.shape[0]):
    temp = np.zeros(3)
    temp[y_test[i]] = 1
    y_test_exp.append(temp)
y_test_exp = np.array(y_test_exp)

y_val_exp = []
for i in range(y_val.shape[0]):
    temp = np.zeros(3)
    temp[y_val[i]] = 1
    y_val_exp.append(temp)
y_val_exp = np.array(y_val_exp)

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications import MobileNetV2

base_model = MobileNetV2(include_top=False, input_shape=(64, 64, 3), weights='imagenet') # weights='imagenet'

# Add less layers
#x = keras.layers.InputLayer(input_shape=(64, 64, 3))
#print(x)
#x = base_model._self_tracked_trackables[1](x)
#x = x.output

# add a global spatial average pooling layer
x = base_model.layers[-57].output # -137 --> First block | -128 --> 2 block |-119-->3block|-110-->4block|-101-->5block|-92-->6block|
#-83->7blocks|-74->8blocks|-65->9blocks
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
#x = Dense(256, activation='relu')(x)
#x = tf.keras.layers.Dropout(rate=0.2)(x)
x = Dense(100, activation='relu')(x)
x = tf.keras.layers.Dropout(rate=0.2)(x)
x = Dense(50, activation='relu')(x)
x = tf.keras.layers.Dropout(rate=0.1)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers[:72]:
    layer.trainable = False

for i, layer in enumerate(model.layers):
   print(i, layer.name)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.8), loss='categorical_crossentropy', metrics=[
      tf.keras.metrics.AUC(name='AUC'),
      tf.keras.metrics.Recall(name='recall')])

# train the model on the new data for a few epochs
hist = model.fit(X_train, y_train_exp,
    epochs=20, 
    validation_data=(X_val, y_val_exp),
    verbose=1)

model.save('model_smote_3.keras')

from operator import itemgetter
import pandas as pd

df = pd.DataFrame(hist.history)
df_2 = df.loc[:, ['AUC', 'val_AUC']]
# Creating a plot
res = df_2.plot(figsize=(8, 6), fontsize=12).get_figure()

# Save figure
res.savefig('plot_ac.png')

df = pd.DataFrame(hist.history)
df_2 = df.loc[:, ['recall', 'val_recall']]
# Creating a plot
res = df_2.plot(figsize=(8, 6), fontsize=12).get_figure()

# Save figure
res.savefig('plot_recall.png')

df = pd.DataFrame(hist.history)
df_2 = df.loc[:, ['loss', 'val_loss']]
# Creating a plot
res = df_2.plot(figsize=(8, 6), fontsize=12).get_figure()

# Save figure
res.savefig('plot_loss.png')



