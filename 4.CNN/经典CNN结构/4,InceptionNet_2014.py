import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dense, GlobalAveragePooling2D

"""
引入了Inception结构块，在同一层网络内使用不同的卷积核，提升了模型感知力
使用了批标准化，缓解了梯度消失
(inception: 开端，创始) 
"""


# 因为所有的小块结构都相同，所以采用类封装以减少代码复杂度
class ConvBNRelu(Model):
    def __init__(self, chanel, kernelSize=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(chanel, kernelSize, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        y = self.model(x, training=False)
        # 在training=False时，BN通过整个训练集计算均值、方差去做批归一化，training=True时，通过当前batch的均值、方差去做批归一化。
        # 推理时 training=False效果好
        return y


# 使用InceptionBlk类用以创建Inception块(InceptionNet核心部件)
class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernelSize=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch, kernelSize=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch, kernelSize=3, strides=1)
        self.c3_1 = ConvBNRelu(ch, kernelSize=1, strides=strides)
        self.c3_2 = ConvBNRelu(ch, kernelSize=5, strides=1)
        self.p4_1 = MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernelSize=1, strides=strides)

    def call(self, x):
        y1 = self.c1(x)
        x2 = self.c2_1(x)
        y2 = self.c2_2(x2)
        x3 = self.c3_1(x)
        y3 = self.c3_2(x3)
        x4 = self.p4_1(x)
        y4 = self.c4_2(x4)
        # concat along axis=channel
        y = tf.concat([y1, y2, y3, y4], axis=3)
        return y


#
class InceptionNet(Model):
    # 参数:num_blocks(Inception块的数量), num_classes(分类的类别数), init_ch(初始通道数)
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        super(InceptionNet, self).__init__(**kwargs)
        self.in_channels = init_ch  # 输入通道量
        self.out_channels = init_ch  # 输出通道量
        self.num_blocks = num_blocks  # 块数
        self.init_ch = init_ch  # 不知道干嘛的
        self.c1 = ConvBNRelu(init_ch)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):  # 块间的循环
            for layer_id in range(2):  # 每个块中的Inception结构块
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)  # 第一个Inception步长2
                else:
                    block = InceptionBlk(self.out_channels, strides=1)  # 第一个Inception步长1
                self.blocks.add(block)  # 创建一个新的InceptionBlk块，并使用add()方法将其添加到blocks序列模型中

            # 在每个块之后，将out_channels乘以2，用于增加下一个块的输出通道数
            self.out_channels *= 2
        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


# 实例化
model = InceptionNet(num_blocks=2, num_classes=10)
