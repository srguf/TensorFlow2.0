import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, \
    GlobalAveragePooling2D
from keras import Model

np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y


model = Baseline()
# 不知道为社么错误
# # 因为所有的小块结构都相同，所以采用类封装以减少代码复杂度
# class ConvBNRelu(Model):
#     def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
#         super(ConvBNRelu, self).__init__()
#         self.model = tf.keras.models.Sequential([
#             Conv2D(ch, kernelsz, strides=strides, padding=padding),
#             BatchNormalization(),
#             Activation('relu')
#         ])
#
#     def call(self, x):
#         y = self.model(x, training=False)
#         # 在training=False时，BN通过整个训练集计算均值、方差去做批归一化，training=True时，通过当前batch的均值、方差去做批归一化。
#         # 推理时 training=False效果好
#         return y
#
#
# class InceptionBlk(Model):
#     def __init__(self, ch, strides=1):
#         super(InceptionBlk, self).__init__()
#         self.ch = ch
#         self.strides = strides
#         self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
#         self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
#         self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
#         self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
#         self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1)
#         self.p4_1 = MaxPool2D(3, strides=1, padding='same')
#         self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides)
#
#     def call(self, x):
#         x1 = self.c1(x)
#         x2_1 = self.c2_1(x)
#         x2_2 = self.c2_2(x2_1)
#         x3_1 = self.c3_1(x)
#         x3_2 = self.c3_2(x3_1)
#         x4_1 = self.p4_1(x)
#         x4_2 = self.c4_2(x4_1)
#         # concat along axis=channel
#         x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
#         return x
#
#
# #
# class InceptionNet(Model):
#     # 参数:num_blocks(Inception块的数量), num_classes(分类的类别数), init_ch(初始通道数)
#     def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
#         super(InceptionNet, self).__init__(**kwargs)
#         self.in_channels = init_ch  # 输入通道量
#         self.out_channels = init_ch  # 输出通道量
#         self.num_blocks = num_blocks  # 块数
#         self.init_ch = init_ch  # 不知道干嘛的
#         self.c1 = ConvBNRelu(init_ch)
#         self.blocks = tf.keras.models.Sequential()
#         for block_id in range(num_blocks):  # 块间的循环
#             for layer_id in range(2):  # 每个块中的Inception结构块
#                 if layer_id == 0:
#                     block = InceptionBlk(self.out_channels, strides=2)  # 第一个Inception步长2
#                 else:
#                     block = InceptionBlk(self.out_channels, strides=1)  # 第一个Inception步长1
#                 self.blocks.add(block)  # 创建一个新的InceptionBlk块，并使用add()方法将其添加到blocks序列模型中
#
#             # 在每个块之后，将out_channels乘以2，用于增加下一个块的输出通道数
#             self.out_channels *= 2
#         self.p1 = GlobalAveragePooling2D()
#         self.f1 = Dense(num_classes, activation='softmax')
#
#     def call(self, x):
#         x = self.c1(x)
#         x = self.blocks(x)
#         x = self.p1(x)
#         y = self.f1(x)
#         return y
#
#
# # 实例化
# model = InceptionNet(num_blocks=2, num_classes=10)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "../../../data/checkpoint/cifar10.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
file = open('../../../data/Con_weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
