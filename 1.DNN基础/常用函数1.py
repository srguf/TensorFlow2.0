import tensorflow as tf

'''
本章知识：
tf.cast(,)
tf.reduce_max(,)  tf.reduce_min()  tf.reduce_mean(,)  tf.reduce_sum(,)
tf.Variable
tf.add(,), tf.subtract(,), tf.multiply(,), tf.divide(,)
tf.Variable()  tf.random.normal(,,)
tf.square(,)  tf.pow(,)   tf.sqrt()  tf.matmul(,)
tf.data.Dataset.from_tensor_slices((,))
'''

x1 = tf.constant([[1., 2., 3.], [4., 5., 6.]], dtype=tf.float64)
print(x1)

# tf.cast
print("********************({})********************".format("cast"))
x2 = tf.cast(x1, tf.int32)  # tf.cast:强制类型转换
print(x2)
print()

# tf.reduce_max(x2):计算张量元素的最大值  tf.reduce_min(x2):计算张量元素的最小值
print("********************({})********************".format("最大最小值"))
print(tf.reduce_max(x2, axis=0), tf.reduce_min(x2))
print()

# tf.reduce_mean(张量名， axis=操作轴):指定维度张量的平均值   tf.reduce_sum(张量名， axis=操作轴):指定维度张量的和
print("********************({})********************".format("平均和求和值"))
print(tf.reduce_mean(x2, axis=0))  # 0竖 1横
print(tf.reduce_sum(x2))
print()

# tf.Variable(初始值)：将变量标记为可训练，被标记的变量会在反向传播中记录梯度信息
print("********************({})********************".format("Variable"))
w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))  # 如此设置在反向传播中就可以通过梯度下降更新参数了
print()

# tf中的数学运算
# 四则运算:tf.add(a,b), tf.subtract(a,b), tf.multiply(a,b), tf.divide(a,b)  #对矩阵来说此处就是对应元素的加减乘除而已
# 平方:tf.square(张量名, n次方数)  次方:tf.pow(张量名, a)   开方:tf.sqrt(a)   #对每个元素分别计算
# 矩阵乘:tf.matmul(矩阵1, 矩阵2)  #正经矩阵运算

# tf.data.Dataset.from_tensor_slices((输入特征, 标签))  #生成输入标签对，生成数据集
'''
注意：
tf.data.Dataset.from_tensor_slices() 函数的参数需要是一个或多个张量（tensor）或 numpy 数组，这些数据会被打包成一个元组。在你的代码中，
x_train 和 y_train 显然是两个不同的数组，所以需要使用元组来将它们打包起来，才能传递给 from_tensor_slices() 函数。

具体而言，在使用 TensorFlow 进行训练时，通常需要将输入数据和标签分别存储在两个数组中。
使用 (x_train, y_train) 的形式将这两个数组打包成一个元组，可以保证它们在训练过程中的对应关系不会发生改变，
且可以方便地一次性传递给 from_tensor_slices() 函数。
'''
print("********************({})********************".format("tf.data.Dataset.from_tensor_slices"))
features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))  # 接收一个元组tensors
print(dataset)
# 细节展示
for element_spec in dataset:
    print(element_spec)
