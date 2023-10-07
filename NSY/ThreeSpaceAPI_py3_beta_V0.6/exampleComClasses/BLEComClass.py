import time
import asyncio
# pip install bleak
from bleak import BleakScanner
from bleak import BleakClient


NOTIFY_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
WRITE_UUID = "0000ffe2-0000-1000-8000-00805f9b34fb"


class BLEComClass:
    def __init__(self, addr=None):
        self.addr = addr
        self.sensor = None
        self.data = bytearray()
        self.data_lock = asyncio.Lock()
        self.loop = asyncio.new_event_loop()

    # Finds 3-Space bluetooth sensor
    async def find_sensor(self):
        device = None
        if self.addr is None:
            # discover() may not find sensor at first - give it a few tries
            num_attempts = 0
            while num_attempts < 3:
                # Find nearby bluetooth devices
                devices = await BleakScanner.discover(service_uuids=["0000ffe0-0000-1000-8000-00805f9b34fb"])  # Some versions of Mac may require an arg for service_uuids
                for d in devices:
                    # Check for 3-Space BLE or MBLE sensors
                    if d.name is not None and "yostlabs" in d.name.lower():
                        self.addr = d.address  # Contains Bluetooth address on Windows, UUID on Mac
                        device = d
                        print("Sensor discovered at address:", self.addr)
                        break
                else:
                    num_attempts += 1
                    continue
                break
            if device is None:
                print("Sensor not discovered")
                exit(0)
        # Create object we will be communicating through
        self.sensor = BleakClient(device)

    # Callback function that is called whenever there is new data to read from the sensor
    async def received_binary_data(self, sender, response):
        # Might not receive all data in one packet, so append to current data
        async with self.data_lock:
            self.data += response

    # async helper method as open() cannot be async (this allows for use of the 'await' keyword)
    async def open_helper(self):
        if self.sensor is None:
            await self.find_sensor()    
        try:
            await self.sensor.connect()
        except:
            print("Error connecting to sensor at address:", self.addr)
            exit(0)
        # This will call the callback function whenever read data changes
        await self.sensor.start_notify(NOTIFY_UUID, self.received_binary_data)
        await asyncio.sleep(1)

    # Required comClass method - must not be async
    def open(self):
        self.loop.run_until_complete(self.open_helper())

    # Required comClass method - must not be async
    def close(self):
        self.loop.run_until_complete(self.sensor.disconnect())

    # async helper method as write() cannot be async (this allows for use of the 'await' keyword)
    async def write_helper(self, data):
        # Write command packet to sensor
        await self.sensor.write_gatt_char(WRITE_UUID, data)

    # Required comClass method - must not be async
    def write(self, data, length):
        asyncio.run(self.write_helper(data))

    # async helper method as read() cannot be async (this allows for use of the 'await' keyword)
    async def read_helper(self, length):
        return_data = bytearray()
        start_time = time.time()
        # Might need to wait for notification callback function to accumulate data
        while time.time() - start_time < 2:
            async with self.data_lock:
                if len(self.data) >= length:
                    return_data = self.data[:length]
                    self.data = self.data[length:]
                    break
            await asyncio.sleep(0.01)
        return return_data

    # Required comClass method - must not be async
    def read(self, length):
        return self.loop.run_until_complete(self.read_helper(length))
