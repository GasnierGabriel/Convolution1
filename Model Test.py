import keras
import cv2
from glob import glob
import numpy as np

model = keras.models.load_model("dogorcat.h5")
nom_image = glob('images/*.jpg')
for image in nom_image:
    image_array = cv2.imread(image, cv2.IMREAD_COLOR)
    image_array = cv2.resize(image_array, (128, 128))
    image_array = image_array.astype('float32')
    image_array /= 255
    image_array = image_array.reshape(1, 128, 128, 3)
    predict = model.predict(image_array)
    predict = np.argmax(predict)

    if predict == 0:
        print("L'image" + image + " est un chat")
    if predict == 1:
        print("L'image" + image + " est un chien")