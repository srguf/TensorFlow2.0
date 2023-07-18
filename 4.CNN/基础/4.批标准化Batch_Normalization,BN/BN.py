"""
tf.keras.layers。BatchNormalization()

示例：
model tf.keras.models.Sequential([
                                Conv2D(filters=6, kernel_size=(5,5), padding='same'),
                                BatchNormalization(),  # BN
                                Activation=('reIu'),  # 激活层
                                MaxPoo12D(pool_size=(2,2), strides=2, padding='same'),
                                Dropout(0.2),  # dropout
                                ])

"""