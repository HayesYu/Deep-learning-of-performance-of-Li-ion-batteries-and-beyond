import pandas as pd

# 读取csv文件，无表头
df = pd.read_csv('/home/user1/code2/id_prop_Al.csv', header=None, names=['id', 'value'])

# 计算均值和标准差
mean = df['value'].mean()
std_dev = df['value'].std()

# 筛选符合条件的行 (均值 ± 2倍标准差范围内)/1.5金属
filtered_df = df[(df['value'] >= (mean - 1.5 * std_dev)) & (df['value'] <= (mean + 1.5 * std_dev))]

# 将结果写入新的csv文件
filtered_df.to_csv('id_prop.csv', index=False, header=False)

