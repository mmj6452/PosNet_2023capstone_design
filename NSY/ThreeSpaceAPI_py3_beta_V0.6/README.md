# ThreeSpaceAPI

The ThreeSpaceAPI is a Python API for the Yost Labs 3-Space Sensors. The goal of the API is to be simple to use and open enough to work on 
any platform that will support Python. This is done through providing a template communication class that is customizable for your needs. 
To keep future maintenence of the API easy we implemented dynamic method generation, once an instance of the ThreeSpaceSensor class has 
been created it will use the provided communication object to detect the type of sensor it is communicating with and generate the methods that 
sensor supports. 

## Requirements

Python >= 3.5

## Usage

```python

from exampleComClasses import USB_ExampleClass
from ThreeSpaceAPI import *

# Create communication object instance.
com = USB_ExampleClass.UsbCom()

# Create sensor instance. This will call the open function of the communication object.
sensor = ThreeSpaceSensor(com)

# Get reading from sensor, returns as tuple.
reading = sensor.getTaredOrientation()

# close communication object and join any spawned threads
sensor.cleanup()

print(reading)
```

## Contributing

The API is still under development, full functionality may be missing for some sensors. Please forward any Errors/Bugs/Feature Requests to techsupport@yostlabs.com

Chain streaming is currently disabled for LX and Nano sensors

## License
[Yost Labs 3-Space Open Source License](https://yostlabs.com/support/open-source-license/)