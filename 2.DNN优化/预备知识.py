import tensorflow as tf
import numpy as np

"""
tf.where(,,)  tf.greater(,)
np.random.RandomState.rand()  np.random.RandomState.rand(,)
np.vstack()
np.mgrid[::,::]  np.c_[,,]  x.ravel()
"""


'''
tf.where(条件语句, 真返回A, 假返回B)

tf.where
tf.greater(a, b): 判断a与b对应元素的比较。
'''
print("********************({})********************".format("tf.where"))
a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
c = tf.where(tf.greater(a, b), a, b)  # 若a>b，则返回a中元素反之b
print(c)
# c = tf.where(a > b, a, b)
# print(c)
print()

'''
np.random.RandomState.rand: 生成一个（0, 1]之间的随机数

np.random.RandomState.rand(维度)
'''
print("********************({})********************".format("np.random.RandomState.rand"))
rdm = np.random.RandomState(seed=1)
a = rdm.rand()
b = rdm.rand(2, 3)  # 返回2x3矩阵
print(a)
print(b)
print()

'''
np.vstack: 两个数组按垂直方向叠加

np.vstack((数组1, 数组2))  注意：vstack函数的参数要求是一个元组，即使你只想堆叠两个数组，也需要将它们放在一个元组中
'''
print("********************({})********************".format("np.vstack"))
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.vstack((a, b))
print(c)

'''
np.mgrid: 创造网格
np.c_: 使返回的间隔数值点配对
x.ravel: 将x变为一维数组

np.mgrid[起始值:结束值:步长, 起始值:结束值:步长, ...]
np.c_[数组1, 数组2, ...]
'''
print("********************({})********************".format("np.mgrid, np.c_, x.ravel()"))
x, y = np.mgrid[1:3:1, 2:4:0.5]  # 创造x, y
grid = np.c_[x.ravel(), y.ravel()]  # 配对
print("x = ", x)
print("y = ", y)
print("grid = ", grid)
