import tensorflow as tf

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()

print(x_train.shape)
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)