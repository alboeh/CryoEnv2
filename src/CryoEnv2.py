import numpy as np

from . import heater
from . import electrothermal as etm
from . import tes

class CryoEnv:
    def __init__(self, record_length, sample_frequency):
        self.record_length = record_length
        self.sample_frequency = sample_frequency
        self.time = np.arange(record_length) / sample_frequency     # time vector
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

        # Now pass all attributes from self.tes to cryo object
        for key, value in vars(self.etm).items():
            setattr(self, key, value)
        print("TES initialized.")

    def get_heater_response(self, V_H):
        I_B = []
        T_e = []
        T_a = []
        if not np.isscalar(V_H):
            for v in V_H:
                _I, _T_e, _T_a = self.etm.solve_equations(self.time, self.tes, self.heater, v)
                I_B.append(_I)
                T_e.append(_T_e)
                T_a.append(_T_a)
        else:
            I_B, T_e, T_a = self.etm.solve_equations(self.time, self.tes, self.heater, V_H)
        return np.array(I_B), np.array(T_e), np.array(T_a)

    ##########
    # HEATER #
    ##########

    def init_heater(self, params: dict):
        self.heater = heater.Heater(**params)
        self.R_H = self.heater.R_H
        print("Heater initialized.")

    ##########
    #  TES   #
    ##########

    def init_tes(self, model: str, params: dict):
        self.tes = tes.TES(model, **params)
        # Now pass all attributes from self.tes to cryo object
        for key, value in vars(self.tes).items():
            setattr(self, key, value)
        print("TES initialized.")

    def get_tes_transition(self, T: float) -> float:
        return self.tes.get_resistance(T)

    def plot_tes_transition(self):
        self.tes.plot_transition()