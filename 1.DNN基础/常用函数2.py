import tensorflow as tf
import numpy as np

"""
本章知识：
tf.GradientTape(,)
enumerate()
tf.one_hot(,)
tf.nn.softmax()
assign_sub()
tf.argmax(,)
"""

'''
tf.GradientTape: with结构记录计算过程，gradient求出张量的梯度

with tf.GradientTape() as tape:
    若干个计算过程
grad = tape.gradient(函数, 对谁求导)
'''
print("********************({})********************".format("tf.GradientTape"))
with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant(3.0))  # w = 3.0
    loss = tf.pow(w, 2)  # 在这里LossFunction被定义为 w平方
grad = tape.gradient(loss, w)  # 返回的是当w = 3时，loss的导数
print(grad)
print()

'''
enumerate: enumerate是Python中的内建函数，它可遍历每个元素(如列表，元组和字符串)，组合为：索引 元素，常在for循环中使用

enumerate(列表名)
'''
print("********************({})********************".format("enumerate"))
seq = ["one", "two", "three"]
# e = enumerate(seq)  # enumerate生成一个对象<enumerate object at 0x0000023FE3969940>
# print(list(e))  # 当您调用list(a)时，Python会遍历a对象，并将其中的所有元素放入一个新的列表中，然后返回该列表对象。
# # 输出: [(0, 'one'), (1, 'two'), (2, 'three')]
for i, element in enumerate(seq):
    print(i, element)
print()

'''
tf.one_hot: 将待转换数据转换为one-hot形式的数据输出

tf.one_hot(待转换数据, depth=几分类)
'''
print("********************({})********************".format("tf.one_hot"))
classes = 3
labels = tf.constant([1, 0, 2])
output = tf.one_hot(labels, depth=classes)  # 3分类
print(output)
print()

'''
tf.nn.softmax: softmax分类器，归一化概率分布

tf.one_hot(x) 相当于对x(注意这里是[n,1])的竖直(好像水平也行)数组)，使输出符合概率分布
'''
print("********************({})********************".format("tf.nn.softmax"))
y = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(y)
print("y_pro is", y_pro)
print()

'''
assign_sub: 赋值操作，跟新参数的值并返回

w.assign_sub(w要减的内容)
'''
print("********************({})********************".format("assign_sub"))
w = tf.Variable(4)
w.assign_sub(1)  # w = w - 1
print(w)
print()

'''
tf.argmax: 返回张量随指定维度最大 索引值

tf.argmax(张量名, axis=操作轴)
'''
print("********************({})********************".format("tf.argmax"))
test = np.array([[1, 2, 3], [2, 3, 4, ], [5, 4, 3, ], [8, 7, 2]])
print(test)
print(tf.argmax(test, axis=0))
print(tf.argmax(test, axis=1))
print()
