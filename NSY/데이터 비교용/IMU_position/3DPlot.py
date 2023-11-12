import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CSV 파일에서 데이터 읽어오기 (파일 경로를 적절히 변경하세요)
data = pd.read_csv('IMU_position/data/3D_plot_building1_ground.csv')

# 3D 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# 데이터에서 x, y, z 값을 추출
x = data['x']
y = data['y']
z = data['z']

print(x)
print(y)
print(z)

# 3D 산점도 그리기
ax.scatter(x, y, z)

# 그래프에 레이블 추가
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 그래프 보여주기
plt.show()