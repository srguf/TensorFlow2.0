import tensorflow as tf
import numpy as np

'''
本章知识：
tf.constant()
tf.zeros()  tf.ones()  tf.fill(,)
tf.random.normal(,,)  tf.random.truncated_normal()
tf.convert_to_tensor()
'''

# 直接创建
print("********************({})********************".format("Tensor的创建"))
a = tf.constant([1, 5], dtype=tf.int64)
print(a)
print(a.dtype)
print(a.shape)
print()

print("********************({})********************".format("Tensor特殊值的创建"))
a1 = tf.zeros([2, 3])
a2 = tf.ones([2, 3])
a3 = tf.fill([2, 2], 9)
print(a1)
print(a2)
print(a3)
# tf.random.normal(维度, mean=均值, stddev=标准差)   生成正态分布的随机数，默认均值为0， 标准差为1
# tf.random.truncated_normal(维度, mean=均值, stddev=标准差)   截断式正态分布的随机数
print()

# numpy转Tensor数据类型
print("********************({})********************".format("numpy转Tensor"))
n = np.array(range(12)).reshape((3, 2, 2))
b = tf.convert_to_tensor(n, dtype=tf.int64)
print(n)
print(b)