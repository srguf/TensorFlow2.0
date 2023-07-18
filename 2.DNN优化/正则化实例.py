import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 读入数据标签，生成x_train, y_train
df = pd.DataFrame(pd.read_csv("./dot.csv"))
#在pandas中，使用单层方括号选取列时，会返回一个Series类型的对象，而使用双层方括号选取多个列时，会返回一个DataFrame类型的对象
#也可以理解为[[]]返回的时二维数组
x_data = np.array(df[["x1", "x2"]])
y_data = np.array(df["y_c"])

x_train = np.vstack(x_data).reshape(-1, 2)
y_train = np.vstack(y_data).reshape(-1, 1)

Y_c = [['red' if y else 'blue'] for y in y_train]  # y_train如果是，则red，否则blue

# 为了防止报错进行数据的格式处理
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

# 生成合集
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

#生成神经网络参数
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

# 训练参数设置
epoch = 500
LR_BASE = 3.99
LR_DECAY = 0.99
LR_STEP = 1
lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
# print(type(epoch))

# 训练部分
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_train, w1) + b1
            h1 = tf.nn.relu(h1)  # 第一层运算
            y = tf.matmul(h1, w2) + b2  # 输出y
            loss_mse = tf.reduce_mean(tf.square(y_train - y))  # mse
            # 添加l2正则化
            loss_regularization = [tf.nn.l2_loss(w1), tf.nn.l2_loss(w2)]
            loss_regularization = tf.reduce_sum(loss_regularization)
            loss = loss_mse + 0.03 * loss_regularization  # 惩罚力度为0.03
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2])

        #实现梯度更新：
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    # 每20个epoch，打印loss信息
    print("epoch = ", epoch, "loss = ", float(loss))

# 预测部分
print("**********predict**********")
# xx在-3到3之间步长为0.01，yy在-3到3之间步长0.01，生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
#将xx，yy拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)

# 将网格坐标点喂入网络进行预测，probs为输出
probs = []
# 将每个坐标点送入神经网络，得到每个点的预测结果
for x_test in grid:
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2
    probs.append(y)

# 取第0列给x1，第一列给x2
x1 = x_data[:, 0]
x2 = x_data[:, 1]
#将probs的shape调整成xx的样子
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c))

plt.contour(xx, yy, probs, levels=[.5])
plt.show()