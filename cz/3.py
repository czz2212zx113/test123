import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
file_path = r"green_tripdata_2016-12.csv"
df = pd.read_csv(file_path)

# 确保日期列正确解析为datetime类型
df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])

# 提取日期部分以用于每日统计
df['date'] = df['lpep_pickup_datetime'].dt.date

# 统计分析：行程距离和车费金额的平均值、最大值、最小值
trip_distance_stats = {
    'mean': df['trip_distance'].mean(),
    'max': df['trip_distance'].max(),
    'min': df['trip_distance'].min(),
}
fare_amount_stats = {
    'mean': df['fare_amount'].mean(),
    'max': df['fare_amount'].max(),
    'min': df['fare_amount'].min(),
}

print("行程距离统计：", trip_distance_stats)
print("车费金额统计：", fare_amount_stats)

# 绘制行程距离和车费金额的直方图
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(df['trip_distance'], bins=30, edgecolor='k')
plt.title('Trip Distance Distribution')
plt.xlabel('Trip Distance')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(df['fare_amount'], bins=30, edgecolor='k')
plt.title('Fare Amount Distribution')
plt.xlabel('Fare Amount')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 周期性分析：每日行程距离和车费金额的变化趋势
daily_stats = df.groupby('date').agg({
    'trip_distance': 'sum',
    'fare_amount': 'sum'
}).reset_index()

# 绘制每日变化趋势的折线图
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(daily_stats['date'], daily_stats['trip_distance'], marker='o', linestyle='-')
plt.title('Daily Trip Distance Trend')
plt.xlabel('Date')
plt.ylabel('Total Trip Distance')
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
plt.plot(daily_stats['date'], daily_stats['fare_amount'], marker='o', linestyle='-')
plt.title('Daily Fare Amount Trend')
plt.xlabel('Date')
plt.ylabel('Total Fare Amount')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
