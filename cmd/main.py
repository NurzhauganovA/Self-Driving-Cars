import tensorflow as tf
import numpy as np


celsius_q = np.array([11, 13, 14, 16, 16, 14, 17], dtype=float)
fahrenheit_a = np.array([51.8, 55.4, 57.2, 60.8, 60.8, 57.4, 62.6], dtype=float)

# for i, c in enumerate(celsius_q):
#     print(f'{c} degrees Celsius = {fahrenheit_a[i]} degrees Fahrenheit')


l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs=5000, verbose=False)
print('Finished training the model')

print(model.predict([20]))