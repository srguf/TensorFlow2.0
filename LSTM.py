import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers


#### 数据处理部分 ####
# 读入数据
stock = pd.read_excel('./道琼斯综合.xls')
stock.tail()  # 查看部分数据
stock.head()

# 时间戳长度
time_stamp = 5  # 输入序列长度

# 划分训练集与验证集
google_stock = stock[['开盘价(元/点)_OpPr']]
train = google_stock[0:7000 + time_stamp]
valid = google_stock[7000 - time_stamp:8500 + time_stamp]
test = google_stock[8500 - time_stamp:]

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train)
x_train, y_train = [], []

# 训练集切片
for i in range(time_stamp, len(train) - 5):
    x_train.append(scaled_data[i - time_stamp:i])
    y_train.append(scaled_data[i: i + 5])

x_train, y_train = np.array(x_train), np.array(y_train).reshape(-1, 5)

# 验证集切片
scaled_data = scaler.fit_transform(valid)
x_valid, y_valid = [], []
for i in range(time_stamp, len(valid) - 5):
    x_valid.append(scaled_data[i - time_stamp:i])
    y_valid.append(scaled_data[i: i + 5])

x_valid, y_valid = np.array(x_valid), np.array(y_valid).reshape(-1, 5)

# 测试集切片
scaled_data = scaler.fit_transform(test)
x_test, y_test = [], []
for i in range(time_stamp, len(test) - 5):
    x_test.append(scaled_data[i - time_stamp:i])
    y_test.append(scaled_data[i: i + 5])

x_test, y_test = np.array(x_test), np.array(y_test).reshape(-1, 5)


#### 建模部分 ####
model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1:])),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dropout(0.1),
    layers.Dense(5)
])

model.compile(optimizer=keras.optimizers.Adam(), loss='mae', metrics=['mae'])
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.7,
                                                            min_lr=0.000000005)
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=70,
                    validation_data=(x_valid, y_valid),
                    callbacks=[learning_rate_reduction])

# loss变化趋势可视化
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.show()


#### 预测结果分析&可视化 ####
closing_price = model.predict(x_test)
model.evaluate(x_test)
scaler.fit_transform(pd.DataFrame(valid['开盘价(元/点)_OpPr'].values))

# 反归一化
closing_price = scaler.inverse_transform(closing_price.reshape(-1, 5)[:, 0].reshape(1, -1))  # 只取第一列
y_test = scaler.inverse_transform(y_test.reshape(-1, 5)[:, 0].reshape(1, -1))

# 计算预测结果
rms = np.sqrt(np.mean(np.power((y_test[0:1, 5:] - closing_price[0:1, 5:]), 2)))
print(rms)
print(closing_price.shape)
print(y_test.shape)

# 预测效果可视化
plt.figure(figsize=(16, 8))
dict_data = {
    'Predictions': closing_price.reshape(1, -1)[0],
    '开盘价(元/点)_OpPr': y_test[0]
}
data_pd = pd.DataFrame(dict_data)
plt.plot(data_pd[['开盘价(元/点)_OpPr']], linewidth=3, alpha=0.8, label='Price')
plt.plot(data_pd[['Predictions']], linewidth=1.2, label='Predict')
plt.legend()
plt.savefig('C:/Users/.guo/Desktop', dpi=600)
plt.show()
