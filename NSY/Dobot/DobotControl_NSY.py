# 23.10.31
# -*- coding: utf-8 -*-

#Data = [time, x, y, z, rHead , acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]

import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import DobotDllType as dType
from lib import USB_ExampleClass
from lib.ThreeSpaceAPI import *

FBpoint = [370,0,-10,0]
RLpoint = [210,200,-10,0]
updownpoint = [210,0,-70,0]

#기준 위치
Home = [210,0,-10,0]
Data = []

#헤드모터가 360도로 동작하지 않기 떄문에 필요한 보정값
angle_bias = 130
#Dobot이 움직일 최소 단위거리
distance = 30
#Dobot이 움직일 횟수
Number_of_repetitions = 3
#Dobot이 움직일 각도 360/angle_divider
angle_divider = 4
#Dobot이 움직일 각도
angle = 260/angle_divider
#데이터 수집 주기
interval = 0.005  # seconds

#변수 셋팅
ts = []
acc_x, acc_y, acc_z = [],[],[]
gyro_x, gyro_y, gyro_z = [],[],[]
mag_x, mag_y, mag_z = [],[],[]

#IMU센서 초기화 및 객체 생성
def init_3space():
    com = USB_ExampleClass.UsbCom()
    print("com: ", com , type(com))
    sensor = ThreeSpaceSensor(com)
    return sensor

#Dobot 초기화 및 객체 생성
def init_Dobot():
    CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
    }
    api = dType.load()
    #Connect Dobot (두봇과 연결 설정)
    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:",CON_STR[state])
    #두봇 연결이 잘되면 초기화
    if (state == dType.DobotConnect.DobotConnect_NoError):
        #Clean Command Queued (명령 큐를 비우기)
        dType.SetQueuedCmdClear(api)
        #Async Motion Params Setting (모션 매개변수 설정)
        dType.SetHOMEParams(api, Home[0], Home[1], Home[2], Home[3], isQueued=1) # Home
        #[설명] dType.SetHOMEParams(api, x, y, z, r, isQueued=1): 홈 위치를 설정
        dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)
        dType.SetPTPCommonParams(api, 100, 100, isQueued = 1)
        dType.SetHOMECmd(api, temp = 0, isQueued = 1)
    return api, state   

#Dobot 종료
def end_Dobot(api):
    dType.SetHOMECmd(api, temp = 0, isQueued = 1)
    dType.SetQueuedCmdStartExec(api)
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
                dType.dSleep(1)
    dType.SetQueuedCmdStopExec(api)
    #Disconnect Dobot (연결 끊기)
    dType.DisconnectDobot(api)

#Dobot 움직이는 함수
def move_Dobot(api, sensor, Home, distance, angle, angle_bias, Data , j , i):
    #+축으로 진행하는 과정
    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, Home[0]+(i*distance), Home[1], Home[2], Home[3]+(j*angle)-angle_bias, isQueued=1)[0]
    dType.SetQueuedCmdStartExec(api)
    #두봇이 움직이는 동안 데이터 수집
    last_time = time.time()
    count = 0
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        if time.time() - last_time > interval * count:
            count += 1
            reading = sensor.getAllRawComponentSensorData()
            x, y, z, rHead, _, _, _, _ = dType.GetPose(api)
            Current_time = time.time()
            Data.append([Current_time , x, y, z, rHead , reading[3], reading[4], reading[5], reading[6], reading[7], reading[8], reading[9], reading[10], reading[11]])
            dType.dSleep(1)
    dType.SetQueuedCmdStopExec(api)
    end_time = time.time()
    print("loop: ", count , "Delay_Time: ", end_time - last_time , "action cycle: ", count/(end_time - last_time))
    
    #-축으로 진행하는 과정
    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, Home[0]+(i), Home[1], Home[2], Home[3]+(j*angle)-angle_bias, isQueued=1)[0]
    dType.SetQueuedCmdStartExec(api)
    last_time = time.time()
    count = 0
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        if time.time() - last_time > interval * count:
            count += 1
            reading = sensor.getAllRawComponentSensorData()
            x, y, z, rHead, _, _, _, _ = dType.GetPose(api)
            Current_time = time.time()
            Data.append([Current_time , x, y, z, rHead , reading[3], reading[4], reading[5], reading[6], reading[7], reading[8], reading[9], reading[10], reading[11]])
            dType.dSleep(1)
    dType.SetQueuedCmdStopExec(api)
    end_time = time.time()
    print("loop: ", count , "Delay_Time: ", end_time - last_time , "action cycle: ", count/(end_time - last_time))
    return Data, lastIndex
def plot(x,y,z,ts):
    plt.subplot(3,1,1)
    plt.plot(ts,x,'r')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.title('x,y,z')
    plt.subplot(3,1,2)
    plt.plot(ts,y,'g')
    plt.xlabel('time')
    plt.ylabel('y')
    plt.subplot(3,1,3)
    plt.plot(ts,z,'b')
    plt.xlabel('time')
    plt.show()
def plot_3D(x,y,z):
    # 3D 그래프 생성
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z)

    # 그래프에 레이블 추가
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # x, y, z 축의 범위를 설정
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.set_zlim(0, 300)

    plt.show()


sensor =  init_3space()
api, state = init_Dobot()

if (state == dType.DobotConnect.DobotConnect_NoError):
################################################ 실질적인 동작 ##############################################################################
    for j in range(angle_divider+1):
        for i in range(Number_of_repetitions+1):
            Data , lastIndex = move_Dobot(api,sensor, Home, distance, angle, angle_bias, Data , j , i)

dType.SetHOMECmd(api, temp = 0, isQueued = 1)
dType.SetQueuedCmdStartExec(api)
while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
            dType.dSleep(1)
dType.SetQueuedCmdStopExec(api)
#Disconnect Dobot (연결 끊기)
dType.DisconnectDobot(api)

#슬라이싱 연산을 위해 numpy 배열로 변환
Data = np.array(Data)

ts= Data[:,0]
x= Data[:,1]
y= Data[:,2]
z= Data[:,3]

plot(x,y,z,ts)
plot_3D(x,y,z)

df = pd.DataFrame(Data, columns = ['time', 'x', 'y', 'z', 'rHead' , 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z'])
df.to_csv('NSY\data/test.csv', index=False, encoding='cp949')

