import tensorflow as tf

model = tf.keras.models.load_model('../models/mymodel.h5')
model.summary()
