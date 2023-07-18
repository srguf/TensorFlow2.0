"""
√TF描述池化
tf.keras.layers.MaxPool2D(
                        pool_size=池化核尺寸，  # 正方形写核长整数，或（核高h,核宽w)
                        strides=池化步长，  # 步长整数，或（纵向步长h,横向步长w),默认为pool size
                        padding=valid'or'same'  # 使用全零填充是“same”,不使用是“valid"”(默认)
                        )

tf.keras.layers.AveragePooling2D(
                                pool_size=池化核尺寸，  # 正方形写核长整数，或（核高，核宽w)
                                strides=池化步长，  # 步长整数，或（纵向步长h,横向步长w),默认为pool _size
                                padding=valid'or'same'  # 使用全零填充是“same”,不使用是“valid”(默认)
                                )

示例：
model tf.keras.models.Sequential([
                                Conv2D(filters=6, kernel_size=(5,5), padding=-'same'),  # 卷积层
                                BatchNormalization(),  # BN
                                Activation=('relu'),  # 激活层
                                MaxPoo12D(pool_size=(2,2), strides=2, padding='same'),  # 池化层
                                Dropout(0.2),  # dropout层
                                ])
"""