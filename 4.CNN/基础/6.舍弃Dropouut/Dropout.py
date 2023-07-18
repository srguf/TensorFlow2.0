"""
tf.keras.layers.Dropout(舍弃的概率)

示例：
model tf.keras.models.Sequential([
                                Conv2D(filters=6, kernel_size=(5,5), padding=-'same'),  # 卷积层
                                BatchNormalization(),  # BN
                                Activation=('relu'),  # 激活层
                                MaxPoo12D(pool_size=(2,2), strides=2, padding='same'),  # 池化层
                                Dropout(0.2),  # dropout层  表示随机舍弃掉20%的神经元
                                ])

卷积是什么？
CBAPD
C      Conv2D(filters=6, kernel_size=(5,5), padding=-'same')
B      BatchNormalization()
A      Activation=('relu')
P      MaxPoo12D(pool_size=(2,2), strides=2, padding='same')
D      Dropout(0.2)
"""