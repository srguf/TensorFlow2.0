"""
model.trainable_variable  返回模型中可训练参数

#设置print输出格式
np.set_printoptions(threshold=超过多少省略显示)

np.set_printoptions(threshold=np.inf)  # np.inf表示无限大

#输出模型中的参数
print(model.trainable_variables)

#存入文本中
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
"""