import tensorflow as tf
import numpy as np
import cv2
import os


class Model:
    def __init__(self):
        pass

    def load_data(self):
        self.train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            '../classification_data/',
            image_size=(224, 224),
            label_mode='categorical'
        )
        self.class_names = self.train_dataset.class_names

    def build(self):
        self.model = tf.keras.applications.MobileNet(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )

        self.model.trainable = False

        self.model = tf.keras.Sequential([
            self.model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(9),
            tf.keras.layers.Softmax()
        ])

    def train(self):
        learning_rate = 0.001
        self.model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        self.model.fit(self.train_dataset, epochs=1)

    def predict(self, path):
        image = cv2.imread(path)
        resize_image = cv2.resize(image, (224, 224))
        data = np.array([resize_image])
        predict = self.model.predict(data)
        index = np.argmax(predict)
        return self.class_names[index]

    def predict_detail(self, path):
        image = cv2.imread(path)
        resize_image = cv2.resize(image, (224, 224))
        data = np.array([resize_image])
        predict = self.model.predict(data)[0]
        result = []
        for i in range(len(predict)):
            result.append((self.class_names[i], predict[i] * 100))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def save(self):
        if not os.path.exists('../models'):
            os.mkdir('../models')
        self.model.save('../models/classification_model_trained.h5')

    def load(self, path):
        self.model = tf.keras.models.load_model(path)


if __name__ == '__main__':
    model = Model()
    model.load_data()
    # model.build()
    # model.train()
    model.load()
    print(model.predict_detail(np.random.rand(1, 2, 3)))
    model.save()
