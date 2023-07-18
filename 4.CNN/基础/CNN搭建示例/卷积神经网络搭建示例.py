from keras import Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense


# 卷积神经网络搭建示例
class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c = Conv2D(filter=6, kernal_size=(5, 5), padding='same')
        self.b = BatchNormalization()
        self.a = Activation('relu')
        self.p = MaxPool2D(pool_size=(2, 2), strids=2, padding='same')
        self.d = Dropout(0.2)

        self.flatten = Flatten()
        self.Dense = Dense(128, activation='relu')
        self.Dropout = Dropout(0.2)
        self.Dense = Dense(10, activation='softmax')

    def recall(self, x):
        x = self.c(x)
        x = self.b(x)
        x = self.a(x)
        x = self.p(x)
        y = self.d(x)
        return y
