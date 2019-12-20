from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# print(tf.__version__)


# імпорт fashion mnist датасету з тенсорфлову
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # load_data поверне 4 масиви numpy
# train_images і test_images масиви numpy розміром 28*28 де кожен елемент [0..255]
# train_labels i test_labels масиви з цілих чисел [0..9] - мітки класу


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# розміри train_images і test_images
# print(train_images.shape)
# print(test_images.shape)


# представлення 17го зображення в тренувальному сеті
# plt.figure()
# plt.imshow(train_images[17])
# plt.colorbar()
# plt.grid(False)
# plt.show()


# масштабування елементів до [0..1] проміжку
train_images = train_images / 255.0
test_images = test_images / 255.0


# можливість проглянути перші 10 елементів
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary) # cmap=plt.cm.binary - чб формат
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


# налаштування моделі
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # вхідний шар на 28 * 28 = 784 пікселів
    keras.layers.Dense(128, activation='relu'), # прихований(обчислювальний) шар на 128 нейронів
    keras.layers.Dense(10, activation='softmax') # вихідний(обчислювальний) шар на 10 нейронів
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # спосіб обчислення похибки
              metrics=['accuracy'])


# навчання моделі
model.fit(train_images, train_labels, epochs=10) # 10 епох тренування


# оцінка точності моделі
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# print('\nТочність: ', test_acc, '\nВтрати: ', test_loss)


# передбачення моделі
predictions = model.predict(test_images)
# print(predictions[3])                                  # масив 10 ймовірностей приналежності до певного класу для екземпляру під 3 номером
# print(np.argmax(predictions[3]))                       # максимальна ймовірність
# print(test_labels[3])                                  # реальне значення мітки
# print(np.argmax(predictions[3]) == test_labels[3])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()