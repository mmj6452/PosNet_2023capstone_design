import struct
import csv

# 이진 파일 이름 및 경로
bin_filename = 'C:/Users/64527/Downloads/Gait-Tracking-With-x-IMU-master/Gait Tracking With x-IMU/Datasets/spiralStairs.bin'
# CSV 파일 이름 및 경로
csv_filename = 'IMU_position\data.csv'

# 이진 파일을 CSV 파일로 변환
csv_data = []
with open(bin_filename, 'rb') as bin_file:
    # 이진 데이터를 읽어오고 변환
    while True:
        binary_data = bin_file.read(4)  # 이 예제에서는 4바이트씩 읽음 (조절 필요)
        if not binary_data:
            break
        # 이진 데이터를 정수로 변환 (이 예제에서는 4바이트 정수로 가정)
        value = struct.unpack('i', binary_data)[0]  # 'i'는 4바이트 정수를 나타냄
        csv_data.append([value])

# CSV 파일로 데이터 쓰기
with open(csv_filename, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(csv_data)

print(f'이진 파일 "{bin_filename}"을 CSV 파일 "{csv_filename}"로 성공적으로 변환했습니다.')
