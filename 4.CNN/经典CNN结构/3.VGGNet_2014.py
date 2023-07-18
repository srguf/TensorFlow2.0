from keras import Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

"""
使用了小尺寸卷积核，在减少了参数的同时，提高了识别的准确率
网咯结构规整，适合硬件加速
"""

class VGGNet(Model):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.c1 = Conv2D(filter=64, kernel_size=(3, 3), strides=1, padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filter=64, kernel_size=(3, 3), strides=1, padding='same')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.d2 = Dropout(0.2)

        self.c3 = Conv2D(filter=128, kernel_size=(3, 3), strides=1, padding='same')
        self.b3 = BatchNormalization()
        self.a3 = Activation('relu')

        self.c4 = Conv2D(filter=128, kernel_size=(3, 3), strides=1, padding='same')
        self.b4 = BatchNormalization()
        self.a4 = Activation('relu')
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.d4 = Dropout(0.2)

        self.c5 = Conv2D(filter=256, kernel_size=(3, 3), strides=1, padding='same')
        self.b5 = BatchNormalization()
        self.a5 = Activation('relu')

        self.c6 = Conv2D(filter=256, kernel_size=(3, 3), strides=1, padding='same')
        self.b6 = BatchNormalization()
        self.a6 = Activation('relu')

        self.c7 = Conv2D(filter=256, kernel_size=(3, 3), strides=1, padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')
        self.p7 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.d7 = Dropout(0.2)

        self.c8 = Conv2D(filter=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b8 = BatchNormalization()
        self.a8 = Activation('relu')

        self.c9 = Conv2D(filter=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b9 = BatchNormalization()
        self.a9 = Activation('relu')

        self.c10 = Conv2D(filter=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')
        self.p10 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.d10 = Dropout(0.2)

        self.c11 = Conv2D(filter=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b11 = BatchNormalization()
        self.a11 = Activation('relu')

        self.c12 = Conv2D(filter=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b12 = BatchNormalization()
        self.a12 = Activation('relu')

        self.c13 = Conv2D(filter=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')
        self.p13 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.d13 = Dropout(0.2)

        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.drop1 = Dropout(0.2)

        self.dense2 = Dense(512, activation='relu')
        self.drop2 = Dropout(0.2)

        self.dense3 = Dense(10, activation='softmax')

    def recall(self, x):
        x = self.c1(x)
        x = self.b1(x)
        y = self.a1(x)
        # 后面的懒得加了
        return y
