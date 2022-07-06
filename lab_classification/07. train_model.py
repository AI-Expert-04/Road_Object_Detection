import tensorflow as tf
import os

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../classification_data/',
    image_size=(224, 224),
    label_mode='categorical'
)

model = tf.keras.models.load_model('../models/mymodel.h5')

learning_rate = 0.001
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    metrics=['accuracy']
)

model.fit(train_dataset, epochs=50)

if not os.path.exists('../models'):
    os.mkdir('../models')

model.save('../models/classification_model_trained.h5')
