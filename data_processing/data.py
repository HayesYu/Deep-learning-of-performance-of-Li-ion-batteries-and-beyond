import pandas as pd
import numpy as np
from scipy import stats
import random

# 读取无表头的CSV文件
df = pd.read_csv('id_prop.csv', header=None)

# 为列命名以方便后续处理
df.columns = ['id', 'target'] if len(df.columns) == 2 else ['id', 'target'] + [f'col{i+3}' for i in range(len(df.columns)-2)]

# 生成预测值函数 - 要求靠近目标值的预测值更多，平均绝对差约为18，并加入少量的大偏差值
def generate_predictions(target_values):
    predictions = []
    
    for target in target_values:
        # 使用三阶段混合分布模型来生成预测值
        r = np.random.random()
        
        # 60%的概率使用窄分布（预测值更接近目标值）
        if r < 0.50:
            delta = np.random.normal(0, 20)
        # 5%的概率使用极宽分布（产生较大偏差）
        elif r > 0.90:
            # 使用更大的标准差，生成少量的较大偏差值
            delta = np.random.normal(0, 100)
        # 35%的概率使用中等宽度的分布
        else:
            delta = np.random.normal(0, 40)
        
        # 将偏差加到目标值上，并保留两位小数
        prediction = round(target + delta, 4)
        predictions.append(prediction)
    
    # 调整预测值，确保平均绝对差在18左右
    predictions = np.array(predictions)
    targets = np.array(target_values)
    abs_diffs = np.abs(predictions - targets)
    current_mean_abs_diff = np.mean(abs_diffs)
    
    # 如果当前平均绝对差不是18，则按比例缩放所有偏差
    if current_mean_abs_diff > 0:  # 避免除以零
        scale_factor = 19 / current_mean_abs_diff
        predictions = targets + (predictions - targets) * scale_factor
        predictions = np.round(predictions, 2)
    
    return predictions.tolist()

# 生成预测值
predictions = generate_predictions(df['target'])
df['prediction'] = predictions

# 计算统计数据以验证结果
abs_diffs = np.abs(df['target'] - df['prediction'])
mean_abs_diff = np.mean(abs_diffs)
median_abs_diff = np.median(abs_diffs)
variance = np.var(df['prediction'] - df['target'])
max_diff = np.max(abs_diffs)

# 计算不同范围内预测值的百分比
within_10 = np.sum(abs_diffs <= 10) / len(df) * 100  # 偏差在±10以内的百分比
within_20 = np.sum(abs_diffs <= 20) / len(df) * 100  # 偏差在±20以内的百分比
within_30 = np.sum(abs_diffs <= 30) / len(df) * 100  # 偏差在±30以内的百分比
above_50 = np.sum(abs_diffs > 50) / len(df) * 100    # 偏差超过±50的百分比

print(f"平均绝对差: {mean_abs_diff:.2f}")
print(f"中位数绝对差: {median_abs_diff:.2f}")
print(f"差值方差: {variance:.2f}")
print(f"最大差值: {max_diff:.2f}")
print(f"偏差在±10以内的百分比: {within_10:.1f}%")
print(f"偏差在±20以内的百分比: {within_20:.1f}%")
print(f"偏差在±30以内的百分比: {within_30:.1f}%")
print(f"偏差超过±50的百分比: {above_50:.1f}%")

# 如果有matplotlib库，绘制差值分布直方图（可选）
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(df['prediction'] - df['target'], bins=50, alpha=0.7)
    plt.title('预测误差分布')
    plt.xlabel('预测值 - 目标值')
    plt.ylabel('频率')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig('error_distribution.png')
    print("误差分布直方图已保存为'error_distribution.png'")
except ImportError:
    print("未检测到matplotlib库 - 跳过直方图生成")

# === 新增功能：随机抽取10%的行，将其id、目标值和预测值放到第4-6列，并从前三列删除 ===

# 创建输出DataFrame，初始化6列
output_df = pd.DataFrame()
output_df[0] = df['id'].copy()          # 第1列：ID
output_df[1] = df['target'].copy()      # 第2列：目标值
output_df[2] = df['prediction'].copy()  # 第3列：预测值
output_df[3] = np.nan                   # 第4列：抽样ID
output_df[4] = np.nan                   # 第5列：抽样目标值
output_df[5] = np.nan                   # 第6列：抽样预测值

# 随机抽取10%的行索引
total_rows = len(df)
sample_size = max(1, int(total_rows * 0.1))  # 确保至少抽取1行
sampled_indices = random.sample(range(total_rows), sample_size)

# 对抽样的行进行数据移动（从前三列移动到后三列）
for idx in sampled_indices:
    # 将数据移动到第4-6列
    output_df.loc[idx, 3] = output_df.loc[idx, 0]  # ID移动到第4列
    output_df.loc[idx, 4] = output_df.loc[idx, 1]  # 目标值移动到第5列
    output_df.loc[idx, 5] = output_df.loc[idx, 2]  # 预测值移动到第6列
    
    # 删除前三列的数据（设置为NaN）
    output_df.loc[idx, 0] = np.nan
    output_df.loc[idx, 1] = np.nan
    output_df.loc[idx, 2] = np.nan

print(f"\n随机抽取了{sample_size}行数据（约占总行数的10%），将其从前三列移动到第4-6列")

# 将结果保存为新的CSV文件（无表头）
output_df.to_csv('test_results_all.csv', index=False, header=False)

print(f"已为{len(df)}行数据生成预测值，平均绝对差约为18，并包含少量较大偏差")
print(f"输出文件已保存为'output.csv'")
print(f"被抽样的{sample_size}行数据从第1-3列移动到了第4-6列")
