import numpy as np

class Heater:
    DEFAULTS = {"R_H": 10e-3,
                "V_H": 0.0} 

    def __init__(self, *, default: bool = False, R_H: float = None, V_H: float = 0.0):
        """
        Heater with optional default initialization.

        Parameters
        ----------
        default : bool, optional
            If True, use built-in defaults (R_H = 10 kΩ).
            If False, user must provide R_H explicitly.
        R_H : float, required if default=False
            Heater resistance in Ohms.
        """
        if default:
            if R_H is not None:
                print("If default=True, R_H will be ignored.")
            self.R_H = self.DEFAULTS["R_H"]
            self.V_H = self.DEFAULTS["V_H"]
        else:
            if not isinstance(R_H, (float, int)) or not isinstance(V_H, (float, int)):
                raise ValueError("If default=False, you must provide valid float or int values for R_H and V_H.")
            self.R_H = float(R_H)
            self.V_H = float(V_H)

        print(f"Heater initialized with R_H = {self.R_H} Ohms")

    def set_heater(self, V_H: float):
        self.V_H = V_H

    def set_heater_tp(self, tp_amplitude: float, record_length: int):
        # create a time-dependent voltage profile for the heater
        self.V_H_tp = self.V_H * np.ones(record_length)
        self.V_H_tp[record_length // 4:int((record_length // 4)*1.1)] += tp_amplitude

    def get_power(self, V_H: float) -> float:
        """
        Calculate heater power given voltage.
        """
        return V_H**2 / self.R_H

    def __repr__(self):
        return f"Heater(R_H={self.R_H:.3g} Ω)"
