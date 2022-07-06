import tensorflow as tf

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../classification_data/',
    image_size=(224, 224),
    label_mode='categorical'
)

data = train_dataset.take(1)
for images, labels in data:
    print(images, labels)

print(train_dataset.class_names)
# 원핫 인코딩
