import tensorflow as tf
import PCAtest as pca
import numpy as np

SAMPLE_SIZE = pca.SAMPLE_SIZE
x_train = pca.feature_array
y_train = np.array(pca.label_encoded_array).reshape(SAMPLE_SIZE,1)

x_test = x_train
y_test = y_train

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(12, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

