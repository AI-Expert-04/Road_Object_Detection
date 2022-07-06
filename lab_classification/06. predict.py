import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('../models/mymodel.h5')

class_names = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'traffic light', 'traffic sign', 'truck']

image = np.random.rand(224, 224, 3)
resize_image = cv2.resize(image, (224, 224))
print(resize_image.shape)

data = np.array([resize_image])
print(data.shape)

predict = model.predict(data)
print(predict)

index = np.argmax(predict)
print(index)

print(class_names[index])
