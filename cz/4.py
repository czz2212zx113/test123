import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. 数据加载与预处理
# 读取数据
df = pd.read_csv('green_tripdata_2016-12.csv')

# 检查数据格式，确保没有缺失值
print(df.info())
print(df.sample(5))

# 转换日期列为datetime格式
df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])

# 处理缺失值：例如，删除缺失出发时间、到达时间、乘客数等的行
df = df.dropna(subset=['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'passenger_count'])

# 计算每个行程的持续时间（分钟）
df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60.0

# 筛选出需要的数据列
df = df[['lpep_pickup_datetime', 'passenger_count', 'trip_duration']]

# 2. 按每10分钟聚合流量数据
df.set_index('lpep_pickup_datetime', inplace=True)

# 按每10分钟进行重采样，计算每个时段内的乘客数量和行程时长总和
df_resampled = df.resample('10T').agg({'passenger_count': 'sum', 'trip_duration': 'sum'})

# 可视化每10分钟的流量数据
plt.figure(figsize=(18, 9))
plt.plot(df_resampled.index, df_resampled['passenger_count'], label='Passenger Count')
plt.xlabel('Date Time')
plt.ylabel('Passenger Count')
plt.title('Traffic Flow (Passenger Count) Every 10 Minutes')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# 保存图像
# plt.savefig('traffic_flow_10min.png')
plt.show()

# 3. 数据分割：将数据分为训练集和测试集
X = df_resampled.index.values.reshape(-1, 1)  # 时间戳作为特征
y = df_resampled['passenger_count'].values  # 目标变量：乘客数量

# 数据分割：将数据分为训练集和测试集
X = df_resampled.index.values.astype('int64') // 10**9  # 转换为 Unix 时间戳（秒数）
X = X.reshape(-1, 1)  # 转换为列向量
y = df_resampled['passenger_count'].values  # 目标变量：乘客数量

# 数据分割：80%训练集，20%测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. 线性回归模型的训练与评估
# 创建线性回归模型
lr_model = LinearRegression()

# 训练模型
lr_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred_lr = lr_model.predict(X_test)

# 计算均方误差（MSE）和决定系数（R²）
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# 输出模型评估指标
print(f'Linear Regression MSE: {mse_lr}')
print(f'Linear Regression R²: {r2_lr}')

# 绘制实际值与预测值的对比图
plt.figure(figsize=(18, 9))
plt.plot(y_test, label='Actual Values')
plt.plot(y_pred_lr, label='Predicted Values', linestyle='--')
plt.xlabel('Test Samples')
plt.ylabel('Passenger Count')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.tight_layout()

# 保存图像
# plt.savefig('lr_actual_vs_predicted.png')
plt.show()

# 5. 决策树回归模型的训练与评估
# 创建决策树回归模型
dt_model = DecisionTreeRegressor(random_state=42)

# 训练模型
dt_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred_dt = dt_model.predict(X_test)

# 计算均方误差（MSE）和决定系数（R²）
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# 输出模型评估指标
print(f'Decision Tree MSE: {mse_dt}')
print(f'Decision Tree R²: {r2_dt}')

# 绘制实际值与预测值的对比图
plt.figure(figsize=(18, 9))
plt.plot(y_test, label='Actual Values')
plt.plot(y_pred_dt, label='Predicted Values', linestyle='--')
plt.xlabel('Test Samples')
plt.ylabel('Passenger Count')
plt.title('Decision Tree: Actual vs Predicted')
plt.legend()
plt.tight_layout()

# 保存图像
# plt.savefig('dt_actual_vs_predicted.png')
plt.show()





# 6. 比较模型性能
# print(f'Linear Regression MSE: {mse_lr}, R²: {r2_lr}')
# print(f'Decision Tree MSE: {mse_dt}, R²: {r2_dt}')
