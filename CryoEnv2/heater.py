import numpy as np

class Heater:
    def __init__(self, resistance):
        if resistance is None:
            raise KeyError("Heater resistance is required. Typical value: 10 mOhm")
        self.resistance = resistance
        self.voltage = 0

    def set_voltage(self, voltage):
        self.voltage = voltage
        return self.voltage
    
    def get_constant_trace(self, record_length, value=None, number: int=1):
        if value is None:
            value = self.voltage
        return np.full((number, record_length), value)
    
    def calc_power_input(self):
        self.power_input = self.voltage ** 2 / self.resistance
        return self.power_input
