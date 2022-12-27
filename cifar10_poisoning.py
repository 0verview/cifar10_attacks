import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Load the CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert the labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the model architecture
input_shape = (32, 32, 3)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the federated learning system
model.fit(x_train, y_train, epochs=10)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')

# Generate poison data
num_poison = 1000
poison_x = np.random.rand(num_poison, 32, 32, 3)
poison_y = np.random.randint(10, size=num_poison)
poison_y = tf.keras.utils.to_categorical(poison_y, num_classes=10)

# Inject the poison data into the training set
x_train = np.concatenate((x_train, poison_x))
y_train = np.concatenate((y_train, poison_y))

# Retrain the model on the poisoned training set
model.fit(x_train, y_train, epochs=10)

# Re-evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy after poisoning: {accuracy:.2f}')
