from sklearn.datasets import load_iris
import pandas as pd

x_data = load_iris().data
y_data = load_iris().target
# print(x_data)  # 一个二维数组
# print(y_data)  # 一个一维数组

x_data = pd.DataFrame(x_data, columns=["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"])
pd.set_option("display.unicode.east_asian_width", True)  # 设置列名对齐
print(x_data)  # [149, 3]相当于150*4矩阵
print()

print("*" * 50)
x_data["类别"] = y_data  # 神奇的python之两部合一步
print(x_data)
print(y_data)


