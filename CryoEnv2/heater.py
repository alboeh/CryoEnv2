import numpy as np

def set_constant_heater(value, number, record_length):
    return np.full((number, record_length), value)
