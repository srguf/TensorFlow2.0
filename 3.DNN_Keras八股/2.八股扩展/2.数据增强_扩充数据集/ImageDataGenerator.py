from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data
x_train, x_test = x_train / 255, x_test / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 因为ImageDataGenerator的fit函数需要4维数据，在这里修改

############################### show new ################################
# 数据增强部分:
image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,  # 缩放图像。（rescale=1. / 255.，分母为255时，可归至0~1）
    rotation_range=45,  # 随机旋转45°
    width_shift_range=.15,  # 宽度偏移±15%
    height_shift_range=.15,  # 高度偏移±15%
    horizontal_flip=True,  # 随机水平翻转
    zoom_range=0.5  # 将图像随机缩放阈量50%(被缩放到原本的50%到150%之间)
)
image_gen_train.fit(x_train)  # 注意：这里四通道，需对数据进行处理
############################### end ################################

# model Sequential
model = tf.keras.models.Sequential(
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
)

#model compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# model fit  注意：这里与标准八股有区别！
model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=5, validation_data=(x_test, y_test),
          validation_freq=1)  # .flow()可以实现动态生成数据批次

model.summary()

# TypeError: cannot unpack non-iterable function object