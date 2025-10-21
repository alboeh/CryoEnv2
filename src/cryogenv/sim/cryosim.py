import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from .core.timebase import init_timebase
from .core.constants import const
from .modules.heater import Heater
from .modules.tes import TES
from .modules.etm import ETM
from .modules.squid import SQUID
from .modules.noise import NOISE

class Cryosim:
    def __init__(self, record_length: int=2**14, sample_frequency=50e3, n_comp=2, set_noise: bool = True, default=True):

        self.record_length = record_length
        self.sample_frequency = sample_frequency
        self.time, self.f, self.w = init_timebase(self, self.record_length, self.sample_frequency)
        self.n_comp = n_comp  # Number of thermal components in the simulation
        self.set_noise = set_noise
        self.modules = {}

        # Give errors for default parameters mode
        if not default:
            print(f"Cryogenic simulation initialized as empty {n_comp}-component simulation. Need to add modules.")
        elif default and n_comp != 2:
            raise ValueError("Default mode only supports n_comp=2. If you want a different number of components, set default=False and add the components manually.")
        elif default:

            # Add Heater
            self.heater = Heater(default=True)
            self.modules['heater'] = self.heater

            # Add TES
            self.tes = TES(default=True)
            self.modules['tes'] = self.tes

            # Add ETM
            self.etm = ETM(n_comp=self.n_comp, default=True)
            self.modules['etm'] = self.etm

            # Add SQUID
            self.squid = SQUID(default=True)
            self.modules['squid'] = self.squid

            # Add noise
            if set_noise:
                self.noise = NOISE(default=True, record_length=self.record_length, sample_frequency=self.sample_frequency)
            
            # Add partices (TODO: implement later)
            self.P_Pe = 0.0  # Pulse power input electrical
            self.P_Pa = 0.0  # Pulse power input absorber

            print("Cryogenic simulation initialized with default components.")
        else:
            raise ValueError("Default parameter must be True or False.")

    def show_modules(self, module=None):        
        # Show all modules or a specific one
        if module:
            return {name: vars(mod) for name, mod in self.modules.items() if name == module}
        else:
            return {name: vars(mod) for name, mod in self.modules.items()}
        
    # Fundamental functions get transferred to cryosim for easy access

    def set_heater(self, heater_voltage: float):
        self.heater.set_heater(heater_voltage)

    def set_heater_tp(self, tp_amplitude: float):
        self.heater.set_heater_tp(time=self.time, tp_amplitude=tp_amplitude)

    def set_bias(self, bias_voltage: float):
        self.etm.set_bias(bias_voltage)

    def solve(self, method="exact", noise: bool = True):
        self.I_T, self.T_e, self.T_a = self.solver(method)

        if self.set_noise:
            self.noise.get_total_noise(self.I_T, T_e=self.T_e, squid=self.squid, tes=self.tes)
            self.squid.get_output(self.I_T_noise)   # Get noise SQUID output
        else:
            self.squid.get_output(self.I_T) # Get V_f in FLL

    def render(self):
        # 2x2 Subplots erstellen
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))

        # Transition curve
        axes[0, 0].plot(np.linspace(25e-3, 45e-3, 500) * 1e3, self.tes.get_resistance(np.linspace(25e-3, 45e-3, 500)) * 1e3)
        axes[0, 0].plot(self.T_e * 1e3, self.tes.get_resistance(self.T_e) * 1e3, 'r.', label='Operating point')
        axes[0, 0].set_title("TES curve")
        axes[0, 0].set_xlabel("Temperature (mK)")
        axes[0, 0].set_ylabel("Resistance (mOhm)")
        axes[0, 0].legend()
        axes[0, 0].grid()

        # Heat transfers
        axes[0, 1].plot(self.time, (self.T_e-self.T_e[0]) * 1e3, label='Delta T_e (electrical) in mK')
        axes[0, 1].plot(self.time, (self.T_a-self.T_a[0]) * 1e3, label='Delta T_a (absorber) in mK')
        axes[0, 1].legend()
        axes[0, 1].set_title("Temperatures")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Temperature (mK)")
        axes[0, 1].grid()

        # Output signal
        axes[1, 0].plot(self.time, (self.V_H + self.V_H_tp) * 1e6, color='orange', label='Heater Voltage')
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Heater Voltage (µV)", color='orange')
        axes[1, 0].tick_params(axis='y', labelcolor='orange')
        axes[1, 0].grid()
        axes[1, 0].set_title("Heater and Feedback Voltages")
        # Secondary axis (right side)
        ax2 = axes[1, 0].twinx()
        ax2.plot(self.time, self.V_f, color='green', label='Feedback Voltage')
        ax2.set_ylabel("Feedback Voltage (V)", color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        # Combine legends from both axes
        lines1, labels1 = axes[1, 0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1, 0].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # NPS
        axes[1, 1].loglog(self.f, self.NPS_flicker, color='blue',linestyle="--", label={"Flicker"})
        axes[1, 1].loglog(self.f, self.NPS_squid, color='blue',linestyle="-.", label={"Squid"})
        axes[1, 1].loglog(self.f, self.NPS_const, color='blue',linestyle=":", label={"Other"})
        axes[1, 1].loglog(self.f, self.NPS, color='green', label={"Overall"})
        axes[1, 1].set_title("Noise Power Spectrum")
        axes[1, 1].set_xlabel("Frequency (Hz)")
        axes[1, 1].set_ylabel("Power (V**2/Hz)")
        axes[1, 1].set_ylim(np.min([self.NPS_squid[1], self.NPS_const[1]])/10,np.max(self.NPS)*10)
        axes[1, 1].grid()
        axes[1, 1].legend()

        # Layout anpassen
        plt.tight_layout()
        plt.show()
    
        
    # IN THIS BLOCK I SOLVE THE ODEs. THIS SHOULD BE LATER MOVED TO THE SEPARATE CORE SOLVER.py

    def rhs(self, t, y, n_comp: int):
        """
        Right-hand side of the ODE system.
        Only for exact numerical solution necessary.

        """
        if self.n_comp != 2:
            raise NotImplementedError("Only n_comp=2 is implemented in ode_solver function.")
        else:
            I_T, T_e, T_a = y

            # Update all time-dependent parameters here if needed
            # For example, if you have a time-dependent heater voltage:
            idx = np.searchsorted(self.time, t)
            idx = np.clip(idx, 0, len(self.time) - 1)   # Makes sure, that no index out of range is taken

            # Get heater voltage in corresponding timestamp
            V_H = self.V_H + self.V_H_tp[idx]  

            R_T = self.tes.get_resistance(T_e)
            P_J = I_T**2 * R_T
            P_H = self.heater.get_power(V_H)

            # Take care of signs!
            dIdt  = (self.V_B - I_T * (self.R_S + R_T)) / self.L
            dTedt = (P_J - self.G_eb * (T_e - self.T_b) - self.G_ea * (T_e - T_a) + P_H + self.P_Pe) / self.C_e
            dTadt = (self.G_ea * (T_e - T_a) - self.G_ab * (T_a - self.T_b) + self.P_Pa) / self.C_a

            dydt = np.array([dIdt, dTedt, dTadt])

        return dydt
    
    def solver(self, method="exact"):

        # Guess starting values (but still give some time to stabilize before t_eval begins)
        self.I_T0 = self.V_B / (2 * self.R_S)
        self.T_e0 = self.tes.T_mid
        self.T_a0 = (self.T_e0 + self.T_b) / 2

        if method == "exact":

            y0 = np.array([self.I_T0, self.T_e0, self.T_a0])  # Starting values

            sol = sp.integrate.solve_ivp(
                fun=lambda t, y: self.rhs(t, y, self.n_comp),    # set ODEs
                t_span=(self.time[0]-0.5, self.time[-1]),         # evaluate over whole time window and a little before, so that the starting values can stabilize
                y0=y0,                                          # starting values
                method="BDF",                                  # "RK23" (Runge-Kutta 2nd order), "RK45" (Runge-Kutta 4th order), "Radau" (implicit Runge-Kutta method), "BDF" (implicit multi-step variable-order method)
                t_eval=self.time,                               # points to give out (only for time values)
                max_step=(self.time[1] - self.time[0]) / 2    # don’t skip over your drive/measurement grid
            )

            I_T, T_e, T_a = sol.y  # Shape: (3, N)
        
        # TODO: Implement fast method

            """
            elif method == "fast":
                self.R_T0 = self.tes.get_resistance(self.T_e0)
                self.alpha = self.T_e0 / self.R_T0 * ((self.tes.get_resistance(self.T_e0+1e-6)-self.tes.get_resistance(self.T_e0-1e-6))/(2e-6))
                self.beta = 0   # TODO: Make TES current dependent
    
                # Get Heater input (also in fourier space)
                dP_H = self.heater.get_power(self.V_H_tp)
                dP_H_f = np.fft.rfft(dP_H)        # fourier transform
    
                # TODO: Implement PileUp particle inputs and noise
                In_f = np.array([np.zeros(len(dP_H_f)), dP_H_f, np.zeros(len(dP_H_f))])
    
                # Create empty solution arrays
                dI_T_f = np.zeros(len(dP_H_f))
                dT_e_f = np.zeros(len(dP_H_f))
                dT_a_f = np.zeros(len(dP_H_f))
    
                for i, w in enumerate(self.w):
                
                    # Define Matrix in Fourier space
                    M = np.array([
                        [1j*w*self.L + self.R_S + self.R_T0*(1 + self.beta),  self.I_T0*self.R_T0*self.alpha/self.T_e0,                      0],
                        [-self.I_T0*self.R_T0*(2 + self.beta),  1j*w*self.C_e - (self.I_T0**2 * self.R_T0 * self.alpha / self.T_e0) + self.G_eb + self.G_ea,  - self.G_ea*self.T_a0/self.T_e0],
                        [0,                             -self.G_ea,                   1j*w*self.C_a + self.G_ab + self.G_ea * self.T_a0 / self.T_e0]
                        ], dtype=complex)
                
                    # Calculate the inverse of the transfer function matrix
                    M_inv = np.linalg.inv(M)
    
                    # Calculate the Fourier transform of the TES current and temperature for the current frequency
                    dI_T_f[i], dT_e_f[i], dT_a_f[i] = M_inv @ In_f[:, i]  # Matrix multiplication for the current frequency
    
                # Get back in real space and add constant values
                I_T = self.I_T0 + np.fft.irfft(dI_T_f)
                T_e = self.T_e0 + np.fft.irfft(dT_e_f)
                T_a = self.T_a0 + np.fft.irfft(dT_a_f)
    
            """

        else:
            raise ValueError("Choose either method 'exact' for numerical solution or 'fast' for fast analytical solution.")
        
        return I_T, T_e, T_a

    # All parameters are generally stored in their respective modules
    # Here we provide easy access via properties also via the cryosim wrapper, so you can directly access them via cryosim.para etc.
    # Besides this, cryosim has no own parameters (except n_comp, record_length, sample_frequency)

    # Heater:
    @property
    def R_H(self): 
        return self.heater.R_H
    
    @property
    def V_H(self): 
        return self.heater.V_H
    
    @property
    def V_H_tp(self): 
        return self.heater.V_H_tp
    
    # TES:
    @property
    def R_T_max(self):
        return self.tes.R_max
    
    # ETM:
    @property
    def C_e(self):
        return self.etm.C_e
    
    @property
    def C_a(self):
        return self.etm.C_a
    
    @property
    def G_eb(self):
        return self.etm.G_eb
    
    @property
    def G_ea(self):
        return self.etm.G_ea
    
    @property
    def G_ab(self):
        return self.etm.G_ab
    
    @property
    def T_b(self):
        return self.etm.T_b
    
    @property
    def R_S(self):
        return self.etm.R_S
    
    @property
    def L(self):
        return self.etm.L
    
    @property
    def V_B(self):
        return self.etm.V_B
    
    # SQUID
    @property
    def V_f(self):
        return self.squid.V_f
    
    @property
    def I_f(self):
        return self.squid.I_f
    
    # NOISE
    @property
    def I_T_noise(self):
        return self.noise.I_T_noise
        
    @property
    def NPS(self):
        return self.noise.NPS
    
    @property
    def NPS_flicker(self):
        return self.noise.NPS_flicker
    
    @property
    def NPS_squid(self):
        return self.noise.NPS_squid
    
    @property
    def NPS_const(self):
        return self.noise.NPS_const
