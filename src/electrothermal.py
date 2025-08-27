import numpy as np
import scipy as sp

class twoComp:
    def __init__(self, **params):
        """
        Explanation of all parameters
        ...
        """
        # Get input values for initial conditions

        # Get inputs
        # TODO: self.lowpass = lowpass
        # TODO: self.noise = noise

        # Get parameters from config
        self.C_e = params["C_e"]    # Heat capacity electrical system
        self.C_a = params["C_a"]    # Heat capacity absorber
        self.G_eb = params["G_eb"]  # Thermal conductance from electrical to bath
        self.G_ea = params["G_ea"]  # Thermal conductance from electrical to absorber
        self.G_ab = params["G_ab"]  # Thermal conductance between absorber and bath
        self.T_b = params["T_b"]    # Temperature of the bath
        self.R_S = params["R_S"]    # Shunt resistance
        self.L = params["L"]        # Inductance of SQUID input coil
        self.V_B = params["V_B"]    # Bias voltage

    def solve_equations(self, time, tes, heater, V_H):  # Solving numerically since this is only option when not using linearizations (we don't want!)

        # Get time values
        self.time = time    # 0 ... 0.32766 s in 20 µs-Schritten
        self.record_length = len(time)
        self.sample_frequency = 1 / (time[1]-time[0])  

        # get classes
        self.tes = tes
        self.heater = heater

        self.V_H = V_H * np.ones(self.record_length) if np.isscalar(V_H) else V_H

        # TODO: Implement radomized heater pulses to imitate Pile-Ups etc. For now on = 0
        self.P_Pe = 0.0; self.P_Pa = 0.0  # Pulse power input

        # Define interpolation functions
        V_H_int = lambda t: np.interp(t, self.time, self.V_H)

        # --- ode ---
        def ODEs(t, y):
            I_T, T_e, T_a = y

            # Import as local variables to speed up
            L = self.L; R_S = self.R_S
            C_e = self.C_e; C_a = self.C_a
            G_eb = self.G_eb; G_ea = self.G_ea; G_ab = self.G_ab
            T_b = self.T_b; 
            P_Pe = self.P_Pe; P_Pa = self.P_Pa
            V_B = self.V_B

            R_T = self.tes.get_resistance(T_e)
            P_J = I_T**2 * R_T
            P_H = self.heater.get_heater_power(V_H_int(t))

            # Take care of signs!
            dI_dt  = (V_B - I_T * (R_S + R_T)) / L
            dTe_dt = (P_J - G_eb*(T_e - T_b) - G_ea*(T_e - T_a) + P_H + P_Pe) / C_e
            dTa_dt = (G_ea*(T_e - T_a) - G_ab*(T_a - T_b) + P_Pa) / C_a

            return np.array([dI_dt, dTe_dt, dTa_dt])

        # Starting values can be only approximated, that leads sometimes to a small step in the beginning
        V_B0 = float(self.V_B if np.isscalar(self.V_B) else self.V_B[0])
        V_H0 = float(self.V_H if np.isscalar(self.V_H) else self.V_H[0])
        P_Pe0 = float(self.P_Pe if np.isscalar(self.P_Pe) else self.P_Pe[0])
        P_Pa0 = float(self.P_Pa if np.isscalar(self.P_Pa) else self.P_Pa[0])
        P_H0 = self.heater.get_heater_power(V_H0)

        I_T0  = V_B0 / (self.R_S + 0.5*self.tes.R_T_max)
        T_e0 = self.T_b + (I_T0 **2 * (0.5*self.tes.R_T_max) + P_H0 + P_Pe0 + (self.G_ea/(self.G_ea+self.G_ab)) * P_Pa0) / (self.G_eb + (self.G_ea*self.G_ab)/(self.G_ea+self.G_ab))
        T_a0 = (self.G_ea*T_e0 + self.G_ab*self.T_b + P_Pa0) / (self.G_ea + self.G_ab)

        y0 = np.array([I_T0, T_e0, T_a0])  # Starting values

        sol = sp.integrate.solve_ivp(
            fun=lambda t, y: ODEs(t, y),    # set ODEs
            t_span=(time[0], time[-1]),         # evaluate over whole time window
            y0=y0,                            # arbitrary starting values
            method="Radau",                     # "Radau" or "BDF"
            t_eval=time,                        # points to evaluate
            #jac=lambda t, y: jac(t, y),         # Optional, to boost calculation
            #atol=[1e-12, 1e-9, 1e-9],         # [A, K, K]
        )

        I_T, T_e, T_a = sol.y[:,-1]  # Shape: (3, N)
        return I_T, T_e, T_a