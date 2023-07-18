"""
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

注释：
plt.subplot() 函数用于创建一个多子图网格，它接受三个参数：plt.subplot(nrows, ncols, index)。
nrows 表示子图的行数。
ncols 表示子图的列数。
index 表示当前子图的索引。
在这段代码中，plt.subplot(1, 2, 1) 创建了一个具有1行、2列的子图网格，并将当前子图设置为索引为1的位置。
也就是说，这段代码将当前绘图区域划分为两个子图，并且当前活动的子图是位于左侧的那个子图（索引从1开始）


"""