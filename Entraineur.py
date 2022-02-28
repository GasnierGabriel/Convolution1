from glob import glob
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import keras

imageSizeX = 1024
imageSizeY = 1024
nbr_batch = 10
nbr_epoch = 10

model = Sequential()

model.add(
        Conv2D(
                32, kernel_size=(3, 3),
                 activation="relu",
                 input_shape=(imageSizeX, imageSizeY, 3)
                 )
          )
model.add(
    MaxPooling2D(pool_size=(3, 3) )
)

model.add(
        Conv2D(
                32, kernel_size=(3, 3),
                 activation="relu"
        )
)
model.add(
    MaxPooling2D(pool_size=(2, 2) )
)

model.add(
    Flatten()
)

model.add(Dense(4096, activation="relu"))
model.add(Dense(4096, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

model.compile(optimizer="adam",
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy']
              )

batch = 1
epoch = 1
while epoch <= nbr_epoch:
        while batch <= nbr_batch:
                x_train = np.load("XTrain" + str(batch) + ".npy")
                y_train = np.load("YTrain" + str(batch) + ".npy")
                model.train_on_batch(x_train, y_train)
                batch += 1
        epoch += 1