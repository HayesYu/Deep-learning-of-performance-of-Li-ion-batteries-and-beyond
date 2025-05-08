import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件，没有表头
csv_file_path = '/home/user1/code/test_results_all.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(csv_file_path, header=None)

# 将列名分别指定为id、target、predict
data.columns = ['id', 'target', 'predict','test_id','test_target','test_predict']

# 提取target和predict列的数据
target = data['target']
predict = data['predict']
test_target=data['test_target']
test_predict=data['test_predict']

# 创建散点图
plt.figure(figsize=(8, 8))
plt.scatter(target, predict, color='blue', label='train', alpha=0.5)
plt.scatter(test_target,test_predict,color='red',label='test',alpha=0.5)

# 添加y=x参考直线
min_val = min(target.min(), predict.min())  # y=x的起点
max_val = max(target.max(), predict.max())  # y=x的终点
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='y = x')

# 添加图例和标签
plt.xlabel('Target')
plt.ylabel('Predict')
plt.title('Target vs Predict')
plt.legend()

# 显示图形
plt.grid(True)
plt.show()

