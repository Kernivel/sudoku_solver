from statistics import mode
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
xTrain = xTrain/255.0
xTest = xTest/255.0
model = tf.keras.Sequential([
    #First layer is the image flattened to a vector
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    #Second layer is the neurons themselves
    tf.keras.layers.Dense(64, activation='relu'),
    #Last layer is goint to be the 10 different classes
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs=10)
test_loss, test_acc = model.evaluate(xTest,  yTest, verbose=2)
print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(xTest)
for i in range(20):
    plt.subplot(5,4,i+1)
    plt.imshow(xTest[i],cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(yTest[i])
    print(yTest[i],"predicted : ",np.argmax(predictions[i]))
plt.show()

model.save("saved_model/MNIST")