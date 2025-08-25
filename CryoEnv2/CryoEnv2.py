import numpy as np

from . import heater

class CryoEnv:
    def __init__(self, recordLength):
        if recordLength is None:
            raise KeyError("Record length is required")
        else:
            self.recordLength = recordLength

    def get_empty_baseline(self, value, number: int = 1):
        if value < 0:
            raise ValueError("Heater value must be non-negative")
        return heater.set_constant_heater(value, number, self.recordLength)
    
    
