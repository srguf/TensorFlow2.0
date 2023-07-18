import tensorflow as tf
import os

x_train, y_train = (), ()
x_test, y_test = (), ()
model = tf.keras.models.Sequential()
model.compile(optimizer='',
              loss='',
              metrics=[])

'''
读取模型:

load_weight(路径文件名)
'''
checkpoint_save_path = "../../../data/checkpoint/mnist.ckpt"  # 文件格式为ckpt
if os.path.exists(checkpoint_save_path + '.index'):  # 生成ckpt文件时会同步生成索引表，所以判断索引表即可知是否保存过模型参数
    print('---------------load the model---------------')
    model.load_weights(checkpoint_save_path)

'''
保存模型:

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=路径文件名,
    save_weights_only=True/False, 是否只保留模型参数
    save_best_only=True/False  是否只保留最优结果    )
history = model.fit(callbacks=[cp_callback])
'''
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint_save_path',
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epoch=5, validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback])  # 与正常的fit相比多了以前的训练数据

model.summary()
