import numpy as np

class Heater:
    def __init__(self, **params):
        if len(params) == 0:
            raise KeyError("Need the following parameters: R_H, parameters given :", params)
        self.R_H = params["R_H"]  # Resistance of the heater in Ohm
    
    def calc_heater_power(self, voltage: float) -> float:
        return voltage ** 2 / self.R_H