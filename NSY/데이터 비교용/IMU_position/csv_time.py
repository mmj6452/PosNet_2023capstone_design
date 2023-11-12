import pandas as pd

# CSV 파일 경로 설정
csv_file1 = 'IMU_position/data/recoded_imu/Accelerometer.csv'
csv_file2 = 'IMU_position/data/recoded_imu/Gyroscope.csv'
csv_file3 = 'IMU_position/data/recoded_imu/Magnetometer.csv'

# CSV 파일을 데이터프레임으로 불러오기
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
df3 = pd.read_csv(csv_file3)

# 필요한 열 추출
columns_to_keep = ['time', 'x', 'y', 'z']

df1 = df1[columns_to_keep].rename(columns={'x': 'x_accelerometer', 'y': 'y_accelerometer', 'z': 'z_accelerometer'})
df2 = df2[columns_to_keep].rename(columns={'x': 'x_gyroscope', 'y': 'y_gyroscope', 'z': 'z_gyroscope'})
df3 = df3[columns_to_keep].rename(columns={'x': 'x_magnetometer', 'y': 'y_magnetometer', 'z': 'z_magnetometer'})

# "time" 열을 기준으로 데이터프레임을 병합
merged_df = pd.merge(df1, df2, on='time', how='inner')
merged_df = pd.merge(merged_df, df3, on='time', how='inner')

# 결과를 CSV 파일로 저장
merged_df.to_csv('output.csv', index=False)

# 결과 출력 (선택 사항)
print(merged_df)
