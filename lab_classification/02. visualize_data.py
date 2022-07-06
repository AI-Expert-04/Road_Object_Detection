import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../classification_data/',
    image_size=(224, 224),
    label_mode='categorical',
)

data = train_dataset.take(1)

plt.figure(0)
plt.title('data')

image_list = []
label_list = []

for images, labels in data:
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(train_dataset.class_names[np.argmax(labels[i])])
        plt.axis('off')
        image_list.append(images[i])
        label_list.append(labels[i])

plt.figure(1)
plt.title('second')
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image_list[i].numpy().astype('uint8'))
    plt.title('predict')
    plt.axis('off')

plt.show()
