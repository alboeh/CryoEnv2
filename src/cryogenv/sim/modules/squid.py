import numpy as np
from ..core.constants import const
from typing import Any, Dict, Iterable

###################
# Starcryo SQUIDs #
###################

class SQUID:

    # central place for defaults
    DEFAULTS = {"M_in": const.Phi0 / 0.44e-6,   # Input coupling (=1/k_in) for SQ300
                "M_f": const.Phi0 / 7.4e-6,   # Feedback coupling FLL (=1/k_f) for SQ300
                "R_f": 100e3}                # Feedback loop resistor

    AVAILABLE_MODELS = ["FLL_simple"]


    """
    SQUID: Model complexity of the SQUID readout.

    Modes
    -----
    simple: Use simplified SQUID model with no jumps, infinite slewrate etc.

    Rules
    -----
    - default=True  -> use built-in defaults for the chosen mode; providing
                      any parameter is NOT allowed (will raise).
    - default=False -> user must provide exactly the required parameters for the mode;
                      missing or extra keys will raise.
    """

    def __init__(self, *, default: bool = False, model: str = "FLL_simple", k_f: float = None, R_f: float = None):
        """
        SQUID model
        ...

        Parameters
        ----------
        ...
        """

        if model != "FLL_simple":
            raise ValueError(f"Unknown model: {model}. Available: {self.AVAILABLE_MODELS}")
        self.model = "FLL_simple"

        if default:
            if R_f is not None or k_f is not None:
                raise TypeError("If default=True, parameters will be ignored.")
            self.R_f = self.DEFAULTS["R_f"]
            self.M_in = self.DEFAULTS["M_in"]
            self.M_f = self.DEFAULTS["M_f"]
        
        print(f"SQUID initialized with model={self.model},M_f = ..., R_f={self.R_f}.")   

    def get_output(self, I_L):
        self.Phi = I_L * self.M_in  # Phi through SQUID
        self.I_f = self.Phi / self.M_f  # Current through feedback coil
        self.V_f = self.R_f * self.I_f  # Output current in feedback coil
        self.V_f *= -1      # Miror output signal on x axis
        self.V_f -= np.median(self.V_f) # Set Baseline to 0
        return self.V_f