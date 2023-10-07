"""
StreamingWireless.py

There are two main ways of using Three Space Sensors: Command and Response,
and Streaming. This example shows how to use Streaming to get samples from a
wireless sensor then log them to a file. This example uses the USB_Example
class but that is able to be swapped with any communication class that works
for your use case.

Setup:
Copy example file into same folder as ThreeSpaceAPI.py and USB_ExampleClass
Connect 3-Space Dongle to PC using usb cable.
Ensure that 3-Space Wireless Sensor serial number is set to logical ID 0 in
Dongle settings.
Power on 3-Space Wireless Sensor.
"""
from exampleComClasses import USB_ExampleClass
from ThreeSpaceAPI import *

# helper function
def hertzToInterval(hertz):
    return int(1000000/hertz)


# Create communication object instance.
com = USB_ExampleClass.UsbCom(timeout=0.05)
# Create sensor instance. This will call the open function of the communication object
# and set our buffer length to desired length.
sensor = ThreeSpaceSensor(com,streamingBufferLen=1000)
# Create log file
logFile = open("exampleLog.txt","w")
# Specify the amount of data points we want.
amountToLog = 500
# Specify the logical ID we want to talk to.
logicalID = 1
# Set sensor to stream Temperature and its orientation
sensor.setStreamingSlots(Streamable.READ_TEMPERATURE_C,Streamable.READ_TARED_ORIENTATION_AS_AXIS_ANGLE,logicalID=logicalID)
# Set sensor to stream at 100Hz for specified number of seconds with no start delay, all arguments are in microseconds
sensor.setStreamingTiming(hertzToInterval(100),STREAM_CONTINUOUSLY ,0 ,logicalID=logicalID)
sensor.setResponseHeaderBitfield(0x53)

print("Starting Streaming!")
sensor.startStreaming(logicalID=logicalID)
data = None
dataPoints = 0
percentage = 0
while dataPoints < amountToLog:
    # Streamed data is added to buffer. This method safely accesses that buffer
    data = sensor.getOldestStreamingPacket(logicalID=logicalID)
    # getOldestStreamingPacket() will return None if there is no data in buffer
    if data is not None:
        # Convert tuple to string, strips parenthesis, and adds new line before writing to file.
        logFile.write(str(data).strip('()') + '\n')
        dataPoints += 1
        # Update and print percentage dots
        curPercentage = int((dataPoints / amountToLog) * 100)
        if curPercentage > percentage:
            diff = curPercentage - percentage
            for i in range(1, diff + 1):
                if (percentage + i) % 10 == 0:
                    print("{}%".format(percentage + i))
                else:
                    print(". ", end='')
            percentage = curPercentage
    else:
        # allow the streaming thread time to fill the streaming buffer
        time.sleep(0)

sensor.stopStreaming(logicalID=logicalID)

# close communication object and join any spawned threads
sensor.cleanup()

logFile.close()
print("Done!")