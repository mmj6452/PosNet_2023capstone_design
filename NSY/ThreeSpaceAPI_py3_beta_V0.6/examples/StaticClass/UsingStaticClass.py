"""
The default setup of the ThreeSpaceAPI is to detect the type of sensor then
dynamically generate the functions that that type of sensor supports. This is
to make maintenance of the API easier and also displays some of the great
features of Python. Since we do generate the code and create the definitions
for the functions when we connect to a sensor it might be slower then what a
production environment would prefer, to solve this we created the
generateStaticClass method, this method will write all the generated functions
and anything else that is necessary to a new python file that will then be
usable as an API for that sensor type. This can also be helpful for debugging
if you are getting unexpected behavior or errors.

In this Example we show you the process for using a static class to do command
and response. The static class was generated using the
GenerateStaticClass.py example.

Setup:
Copy example file into same folder as ThreeSpaceAPI.py and USB_ExampleClass
Connect 3-Space Sensor to PC using usb cable.
"""
from exampleComClasses import USB_ExampleClass
from ExampleStaticAPI import *

# Create communication object instance.
com = USB_ExampleClass.UsbCom()
# Create sensor instance. This will call the open function of the communication object.
sensor = ThreeSpaceSensor(com)
# Create list to hold data
dataList = []
# Specify the amount of data points we want.
amountToLog = 250
print("Starting data collection!")
for i in range(amountToLog):
    # Get reading from sensor, returns as tuple.
    reading = sensor.getTaredOrientation()
    # save reading to data list
    dataList.append(reading)
# close communication object and join any spawned threads
sensor.cleanup()
print("Logging data to file!")
# Create log file
logFile = open("exampleLog.txt","w")
for data in dataList:
    # Convert tuple to string, strips parenthesis, and adds new line before writing to file.
    logFile.write(str(data).strip('()') + '\n')

logFile.close()
print("Done!")
