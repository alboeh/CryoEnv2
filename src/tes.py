import numpy as np
import matplotlib.pyplot as plt

class TES:
    def __init__(self, model: str, **params):
        self.model = model
        if self.model not in ["sigmoid"]:
            raise ValueError("Unknown model: {}".format(model), "available models are: ['sigmoid']")
        if len(params) == 0:
            raise KeyError("Need the following parameters: R_T_max, T_c, parameters given :", params)
        self.R_T_max = params["R_T_max"]  # Maximum resistance of the TES in Ohm
        self.T_min = params["T_min"]  # Minimum temperature of the TES in K
        self.T_c = params["T_c"]  # Critical temperature of the TES in K
        self.T_max = params["T_max"]  # Maximum temperature of the TES in K

    def get_resistance(self, T: float) -> np.ndarray | float:
        """Calculate the resistance of the TES based on its temperature using a sigmoid model."""
        T = np.asarray(T)   # So that it can take single floats or arrays
        if self.model == "sigmoid":
            R = self.R_T_max / (1 + np.exp(-10 * (T - self.T_c) / (self.T_max - self.T_min)))
            return R
        else:
            raise ValueError("Unknown model: {}".format(self.model), "available models are: ['sigmoid']")
        
    ########### PLOTS ############

    def plot_transition(self):
        plt.figure(figsize=(8, 5))
        plt.plot(np.linspace(0, (self.T_max+0.05)*1e3, 1000), self.get_resistance(np.linspace(0, self.T_max+0.05, 1000))*1e3)
        plt.title("TES Transition Curve")
        plt.xlabel("Temperature (mK)")
        plt.ylabel("Resistance (mOhm)")
        plt.grid()
        plt.show()