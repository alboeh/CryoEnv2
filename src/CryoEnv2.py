import numpy as np

from . import heater
from . import electrothermal as etm

class CryoEnv:
    def __init__(self, record_length, sample_frequency):
        self.record_length = record_length
        self.sample_frequency = sample_frequency
        print("Cryogenic environment initialized.")

    ##########
    #  ETM   #
    ##########

    def init_etm(self, n_components: int, params: dict):
        if n_components == 2:
            self.etm = etm.twoComp(**params)
            print("2-component Electrothermal model initialized.")
        else: 
            KeyError("Only 2-component model implemented yet.")

    ##########
    # HEATER #
    ##########

    def init_heater(self, params: dict):
        self.heater = heater.Heater(**params)
        self.R_H = self.heater.R_H
        print("Heater initialized.")

    def set_heater_voltage(self, voltage: float):
        self.V_H = voltage
        self.P_H = self.heater.calc_heater_power(self.V_H)
        print(f"Heater voltage set to {voltage} V.")