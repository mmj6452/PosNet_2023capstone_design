""" 
    YOSTLABS 3-SPACE SENSOR PYTHON EXAMPLE CODE
    읽어드린 IMU데이터를 CSV파일로 저장하는 코드
    time_interval을 조정하여 데이터를 읽어오는 속도를 조정할 수 있음
    duration을 조정하여 일정 시간 동안 데이터를 읽어오는 것도 가능 (단위: 초)
    duration을 0으로 설정하면 무한정 데이터를 읽어옴
    PC USB포트에 IMU를 연결하고 COM port를 지정해주면 실행
"""

from exampleComClasses import USB_ExampleClass
from ThreeSpaceAPI import *
import time
import csv

# 인터벌 시간을 설정하여 Hz 조정
time_interval = 0.005  # 200Hz로 데이터 읽기

# 일정 시간 동안 데이터를 읽어오려면 아래 변수 설정
duration = 60  # 60초 동안 데이터를 읽어옴 (예: 1분)

# Create communication object instance.
com = USB_ExampleClass.UsbCom()

# Create sensor instance. This will call the open function of the communication object.
sensor = ThreeSpaceSensor(com)

# Create and open a CSV file for writing
with open('sensor_data.csv', 'w', newline='') as csvfile:
    # Create a CSV writer
    csvwriter = csv.writer(csvfile)

    # Write a header row
    csvwriter.writerow(['Timestamp', 'Value1', 'Value2', 'Value3', 'Value4', 'Value5', 'Value6', 'Value7', 'Value8', 'Value9', 'Value10', 'Value11', 'Value12'])

    start_time = time.time()
    end_time = start_time + duration if duration > 0 else None

    while end_time is None or time.time() < end_time:
        # Get reading from sensor, returns as tuple.
        reading = sensor.getAllCorrectedComponentSensorData()
        print(reading)
        # Write the reading to the CSV file
        csvwriter.writerow(reading)
        print("Data written to CSV file")

        time.sleep(time_interval)

# close communication object and join any spawned threads
sensor.cleanup()