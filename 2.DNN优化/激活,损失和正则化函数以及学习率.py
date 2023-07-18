import tensorflow as tf

"""
激活函数：
#Sigmoid函数
tf.nn.sigmoid()

#Tanh函数
tf.math.tanh()

#Relu
tf.nn.relu()

#Leaky Relu
tf.nn.leaky_relu()
"""

"""
损失函数：
y_: 标准答案
y: 计算结果

均方损失函数： mse(Mean Squared Error)
公式： tf.reduce_mean(tf.square(y_ - y))
tf.losses.mean_squared_error()

交叉熵损失函数：CE(Cross Entropy)    
公式：-(y_ * lny)的加和
tf.losses.categorical_crossentropy(y_, y) 

将softmax与交叉熵结合
y_pro = tf.nn.softmax(y)
loss1 = tf.losses.categorical_crossentropy(y_, y_pro) 

loss2 = tf.nn.softmax_cross_entropy_with_logits(y_, y)  loss2 = y_pro + loss1
"""

"""
正则化函数：
tf.nn.l2_loss[w1]

具体实现：
loss_regularization = []
loss_regularization.append(tf.nn.l2_loss(w1))
loss_regularization.append(tf.nn.l2_loss(w2))
...
loss_regularization = tf.reduce_sum(loss_regularization)
"""

"""
指数衰减学习率：
指数衰减学习率 = 初始学习率 * 学习率衰减率^(当前轮数 / 多少轮衰减一次)

示例
epoch = 40
LR_BASE = 0.2
LR_DECAY = 0.99
LR_STEP = 1

lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
"""