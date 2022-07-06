import tensorflow as tf
import numpy as np
import cv2
import os

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../classification_data/',
    image_size=(224, 224),
    label_mode='categorical'
)

class_names = train_dataset.class_names

model = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

model.trainable = False

model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(9),
    tf.keras.layers.Softmax()
])

learning_rate = 0.001
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    metrics=['accuracy']
)

model.fit(train_dataset, epochs=30)

if not os.path.exists('../models'):
    os.mkdir('../models')

model.save('../models/classification_model_trained.h5')

class_name = input()
file_name = input()

image = cv2.imread('../classification_data/' + class_name + '/' + file_name)
resize_image = cv2.resize(image, (224, 224))

data = np.array([resize_image])

predict = model.predict(data)
print(predict)

index = np.argmax(predict)
print(index)

print(class_names[index])

cv2.imshow('image', image)
cv2.waitKey(0)
