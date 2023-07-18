import tensorflow as tf

"""
one-hot: 数据量大，过于稀疏，映射之间相互独立，没有表现出关联性。
Embedding: 是一种单词编码方法，用低维向量实现了编码。
        这种编码通过神经网络的优化，可以表现出单词间的相关性
        
tf.keras.layers.Embedding(词汇表大小, 编码维度)  编码维度就是用几个数字表达一个单词

例：tf.keras.layers.Embedding(100，3)    对1-100进行编码，   


注意：当使用Embedding时，x_train维度：
[送入样本数，循环核时间展开步数]
x_train = np.reshape(x_train, (len(x_train), 1))
"""

