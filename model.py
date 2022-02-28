from glob import glob
from cv2 import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


dossierTest = glob("test/*")
dossierTrain = glob("train/*")
label = 0

x_train = []
y_train = []

x_test = []
y_test = []

for dossier in dossierTrain:
    label = dossier[-1]
    photos = glob(dossier + "/*.jpg")
    for photo in photos:
        image = cv2.imread(photo, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (200, 200))
        image = image.astype('float32') / 255
        x_train.append(image)
        y_train.append(label)

for dossier in dossierTest:
    label = dossier[-1]
    photos = glob(dossier + "/*.jpg")
    for photo in photos:
        image = cv2.imread(photo, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (200, 200))
        image = image.astype('float32') / 255
        x_test.append(image)
        y_test.append(label)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

model = Sequential()

model.add(Conv2D(64, kernel_size=(4, 4),
                 activation ='relu',
                 input_shape=(200, 200, 3)))

model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(128, kernel_size=(4, 4),
                 activation ='relu'))

model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(256, kernel_size=(4, 4),
                 activation ='relu'))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer="Adam",
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=10, batch_size=65, validation_data=(x_test, y_test))
model.save('dogorcat.h5')


