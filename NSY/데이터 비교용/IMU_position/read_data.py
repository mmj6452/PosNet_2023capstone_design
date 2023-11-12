import pandas as pd

path = "IMU_position/building1/known/0.feather"
df = pd.read_feather(path)
print(df)

output = df.to_csv("IMU_position/building1/known/csv/0.csv", index=False)


