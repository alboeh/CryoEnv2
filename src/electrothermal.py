import numpy as np

class twoComp:
    def __init__(self, **params):
        # Get input values for initial conditions
        """
        self.t = t
        self.N = len(t)
        self.N_f = self.N // 2 + 1      # Frequency space has less points because of real rfft trafo
        self.dt = t[1] - t[0]
        self.fs = 1 / self.dt   # Sampling frequency
        self.f = sp.fft.rfftfreq(self.N, d=self.dt)
        self.w = 2 * np.pi * self.f   # for noise calculations only positive frequencies are possible

        # Get inputs
        self.antialiasing = antialiasing
        self.noise = noise
        self.modulation = modulation
        self.input = inputModel
        self.solver = solver

        # Get parameters from config
        self.params = params
        self.alpha = params.alpha
        self.beta = params.beta
        self.T_c = params.T_c
        self.T_ac = params.T_ac
        self.T_b = params.T_b
        self.R_c = params.R_c
        self.R_S = params.R_S
        self.C = params.C
        self.C_a = params.C_a
        self.G_tb = params.G_tb
        self.G_ta = params.G_ta
        self.G_ab = params.G_ab
        self.L = params.L

        # Get biasing parameters
        self.I_e = params.I_e
        self.f_0 = params.f_0
        self.w_0 = 2 * np.pi * self.f_0
        self.kap = params.kap

        # Calc all values for ETM model
        self.calcValues()

    def calcValues(self):    # Master function to do all the calculations in right order

        self.DP = self.input.DPe  # Input power
        self.DP_a = self.input.DPa  # Input power for the absorber component
        self.DP_f = sp.fft.rfft(self.DP)  # Fourier transform of the input power
        self.DP_a_f = sp.fft.rfft(self.DP_a)

        # No noise should be applied for now, so this is 0
        self.DV = np.zeros(self.N)  # Voltage change by noise (not including modulation)
        self.DV_f = sp.fft.rfft(self.DV)  # Fourier transform of the input voltage

        self.solveODEs()    # Calculate the TES temperature change due to input

    def setPar(self, **kwargs):     # Function to set parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                raise AttributeError(f"Parameter '{key}' not found in params of ETM")
        
        # Recalculate dependent values after updating parameters
        self.calcValues()
    
    def solveODEs(self):   # Li is not neglibible (TODO: Put in separate module TES.py)

        # Initialize arrays for Fourier transform of TES current and temperature
        self.DI_f = np.zeros(self.N_f, dtype=complex)
        self.DT_f = np.zeros(self.N_f, dtype=complex)
        self.DT_a_f = np.zeros(self.N_f, dtype=complex)

        u_f = np.array([self.DV_f, self.DP_f, self.DP_a_f])  # Input array
        # Loop through each frequency step
        for i, w in enumerate(self.w):
            M = np.array([  # Transfer function matrix for the current frequency
                [1j * w * self.L + self.R_S + self.R_c * (1 + self.beta),
                    self.I_e * self.R_c * self.alpha / self.T_c,
                    0],
                [-self.I_e * self.R_c * (2 + self.beta * (1 + self.kap / 2)),
                    1j * w * self.C - self.I_e**2 * self.R_c * self.alpha / self.T_c * (1 + self.kap / 2) + self.G_tb + self.G_ta,
                    -self.G_ab * self.T_ac / self.T_c],
                [0,
                    -self.G_ta,
                    1j * w * self.C_a + self.G_ab + self.G_ta * self.T_ac / self.T_c]
            ], dtype=complex)
            # Calculate the inverse of the transfer function matrix
            M_inv = np.linalg.inv(M)
            # Calculate the Fourier transform of the TES current and temperature for the current frequency
            self.DI_f[i], self.DT_f[i], self.DT_a_f[i] = M_inv @ u_f[:, i]  # Matrix multiplication for the current frequency
        self.DI = sp.fft.irfft(self.DI_f)
        self.DT = sp.fft.irfft(self.DT_f)
        self.DT_a = sp.fft.irfft(self.DT_a_f)

        self.I = np.sqrt(2)*(self.I_e + self.DI)*np.sin(self.w_0*self.t)  # TES current
        self.T = self.DT + self.T_c   # TES temperature by adding the change to the critical temperature
        self.T_a = self.DT_a + self.T_ac
        
        # Calc the modulation index m for AM modulation
        #if self.modulation == "AM":
        #    self.m = (np.m)
        """