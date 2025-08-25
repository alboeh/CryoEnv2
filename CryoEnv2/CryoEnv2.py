import numpy as np

from . import heater

class CryoEnv:
    def __init__(self, record_length):
        self.record_length = record_length

    def init_heater(self, resistance):
        self.heater = heater.Heater(resistance)
        print("Heater initialized.")

    def set_heater_voltage(self, voltage):
        if isinstance(self.heater, type(None)):
            raise KeyError("Heater not initialized. Call init_heater(resistance) first.")
        self.heater_voltage = self.heater.set_voltage(voltage)
        return self.heater_voltage

    def get_constant_heater_trace(self, voltage=None, number: int=1):
        self.heater.calc_power_input()   # Calc the power input
        return self.heater.get_constant_trace(self.record_length, voltage, number)

    def get_testpulse_trace(self):
        ...
