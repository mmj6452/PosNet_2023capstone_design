import csv
import struct

# CSV 파일 이름 및 경로
csv_filename = 'C:/Users/64527/Downloads/Gait-Tracking-With-x-IMU-master/Gait Tracking With x-IMU/Datasets/test_NSY_home_CalInertialAndMag.csv'
# 이진 파일 이름 및 경로
bin_filename = 'C:/Users/64527/Downloads/Gait-Tracking-With-x-IMU-master/Gait Tracking With x-IMU/Datasets/test_NSY_home.bin'

# CSV 파일 읽기
csv_data = []
with open(csv_filename, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    # 첫 번째 줄은 헤더이므로 건너뜁니다.
    next(csv_reader)
    for row in csv_reader:
        csv_data.append(row)

# 이진 파일로 데이터 쓰기
with open(bin_filename, 'wb') as bin_file:
    for row in csv_data:
        for item in row:
            try:
                # 빈 문자열이면 0.0으로 처리하고 이진 형식으로 저장
                if item.strip() == '':
                    binary_data = struct.pack('f', 0.0)
                else:
                    binary_data = struct.pack('f', float(item))
                bin_file.write(binary_data)
            except ValueError:
                # 부동 소수점으로 변환할 수 없는 경우 예외 처리
                print(f"Could not convert to float: {item}")

print(f'CSV 파일 "{csv_filename}"을 이진 파일 "{bin_filename}"로 성공적으로 변환했습니다.')

