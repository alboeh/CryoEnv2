import numpy as np

from numpy.typing import ArrayLike  # For having floats as well as np.ndarrays in function arguments

class TES:
    # central place for defaults
    DEFAULTS = {"R_max": 150e-3, 
                "T_mid": 35e-3}

    AVAILABLE_MODELS = ["sigmoid"]

    def __init__(self, *, default: bool = False, model: str = "sigmoid", R_max: float = None, T_mid: float = None):
        """
        Transition Edge Sensor (TES) resistance model.
        Currently only sigmoid model is supported.

        Parameters
        ----------
        default : bool, optional
            If True, use built-in default parameters.
            If False, user must provide R_max and T_mid.
        R_max : float, required if default=False
            Maximum resistance of the TES in Ohm.
        T_mid : float, required if default=False
            Midpoint temperature of the TES transition (K).
        """

        if model != "sigmoid":
            raise ValueError(f"Unknown model: {model}. Available: {self.AVAILABLE_MODELS}")
        self.model = "sigmoid"

        if default:
            if R_max is not None or T_mid is not None:
                raise TypeError("If default=True, parameters will be ignored.")
            self.R_max = self.DEFAULTS["R_max"]
            self.T_mid = self.DEFAULTS["T_mid"]
        
        else:
            if R_max is None or T_mid is None:
                raise KeyError("If default=False, you must provide R_max and T_mid.")
            self.R_max = float(R_max)
            self.T_mid = float(T_mid)
        
        print(f"TES initialized with model={self.model}, R_max={self.R_max} Ohm, T_mid={self.T_mid} K")   

    def get_resistance(self, T: ArrayLike) -> ArrayLike:
        """
        Smooth s-curve:
        R(T) ≈ 0 for T << T_mid
        R(T) ≈ R_max for T >> T_mid
        """
        T = np.asarray(T, dtype=float)
        if self.model == "sigmoid":
            slope = 200.0  # Steepness of the transition
            x = (T - self.T_mid) * slope
            R = self.R_max / (1.0 + np.exp(-x))
            return R
        else:
            raise ValueError(
                f"Unknown model: {self.model}. Available: ['sigmoid']"
        )