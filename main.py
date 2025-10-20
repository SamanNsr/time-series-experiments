import pandas as pd
import numpy as np
from data_loader import TimeSeriesBatchGenerator




df = pd.read_csv('./datasets/electricity/electricity.csv', parse_dates=['date'])


df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

data = df.drop(columns=["date", "OT"]).values

print(data[:5])

train_ratio = 0.8 
T = data.shape[0]
print("Total time steps:", T)
split_idx = int(T * train_ratio)

train_data = data[:split_idx] 
test_data  = data[split_idx:]  

window_size = 24
horizon = 4
batch_size = 16
train_gen = TimeSeriesBatchGenerator(train_data, window_size, horizon, batch_size, shuffle=True, drop_last=False)
for i, (X_batch, Y_batch) in enumerate(train_gen):
    print(i)
#     print(f"Batch {i+1}:")
#     print("X_batch shape:", X_batch.shape)
#     print("Y_batch shape:", Y_batch.shape)


test_gen = TimeSeriesBatchGenerator(test_data, window_size, horizon, batch_size, shuffle=False, drop_last=False)

for i, (X_batch, Y_batch) in enumerate(test_gen):

    print(f"Test Sample {i+1}:")
#     print("X_batch shape:", X_batch.shape)
#     print("Y_batch shape:", Y_batch.shape)
#     # if i == 2:  
#     #     break

