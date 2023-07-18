"""import tensorflow as tf

tf.keras.layers.SimpleRNN(1,  # 记忆体个数
                          activation="relu",  # 不写默认是tanh
                          return_sequences=True, # True:各时间步输出隐层 False:仅最后时间步输出隐层
                          )

# 入RNN时，x_train维度: [送入样本数， 循环核时间展开步数， 每个时间输入特征个数]
                        几组          几批              每组几个

例：
0.4, 1.7, 0.6
0.7, 0.9, 1.6
RNN期待维度：[2, 1, 3]

---------------------------------->时间轴
0.4, 1.7    0.2, 1.7    0.1, 1.1    1.1, 0.1
[1, 4, 2]
"""