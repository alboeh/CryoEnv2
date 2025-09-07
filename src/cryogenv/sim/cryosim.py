import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from .core.timebase import init_timebase
from .modules.heater import Heater
from .modules.tes import TES
from .modules.etm import ETM

class Cryosim:
    def __init__(self, record_length=2**14, sample_frequency=50e3, n_comp=2, default=True):

        self.record_length = record_length
        self.sample_frequency = sample_frequency
        init_timebase(self, self.record_length, self.sample_frequency)
        self.n_comp = n_comp  # Number of thermal components in the simulation
        self.modules = {}

        # Give errors for default parameters mode
        if not default:
            print(f"Cryogenic simulation initialized as empty {n_comp}-component simulation. Need so add modules.")
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

    def set_heater(self, V_H: float):
        self.heater.set_heater(V_H)

    def set_heater_tp(self, tp_amplitude: float):
        self.heater.set_heater_tp(tp_amplitude, self.record_length)
        self.I_T, self.T_e, self.T_a = self.solver()

    def render(self):
        # 2x2 Subplots erstellen
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Einzelne Achsen ansprechen
        axes[0, 0].plot(np.linspace(0, self.R_T_max*2, 100), self.tes.get_resistance(np.linspace(0, self.R_T_max*2, 100)))
        axes[0, 0].plot(self.T_e, self.tes.get_resistance(self.T_e), 'r.', label='Operating point')
        axes[0, 0].set_title("TES curve")

        axes[0, 1].plot(self.time, (self.T_e-self.T_e[0]) * 1e3, label='Delta T_e (electrical) in mK')
        axes[0, 1].plot(self.time, (self.T_a-self.T_a[0]) * 1e3, label='Delta T_a (absorber) in mK')
        axes[0, 1].legend()
        axes[0, 1].set_title("Temperatures")

        axes[1, 0].plot(self.time, self.V_H_tp, color='orange', label='Heater Voltage')
        axes[1, 0].set_title("Heater voltage")

        axes[1, 1].plot(self.time, self.I_T)
        axes[1, 1].set_title("Current through TES")

        # Layout anpassen
        plt.tight_layout()
        plt.show()
    
        
    # IN THIS BLOCK I SOLVE THE ODEs. THIS SHOULD BE LATER MOVED TO THE SEPARATE CORE SOLVER.py

    def rhs(self, t, y, n_comp: int):
        """
        Right-hand side of the ODE system.

        """
        if self.n_comp != 2:
            raise NotImplementedError("Only n_comp=2 is implemented in ode_solver function.")
        else:
            I_T, T_e, T_a = y

            # Update all time-dependent parameters here if needed
            # For example, if you have a time-dependent heater voltage:
            if hasattr(self.heater, 'V_H_tp'):
                idx = np.searchsorted(self.time, t)
                if 0 <= idx < len(self.heater.V_H_tp):
                    V_H = self.heater.V_H_tp[idx]
            else:
                V_H = self.V_H  # constant voltage if no time profile is set

            R_T = self.tes.get_resistance(T_e)
            P_J = I_T**2 * R_T
            P_H = self.heater.get_power(V_H)

            # Take care of signs!
            dIdt  = (self.V_B - I_T * (self.R_S + R_T)) / self.L
            dTedt = (P_J - self.G_eb * (T_e - self.T_b) - self.G_ea * (T_e - T_a) + P_H + self.P_Pe) / self.C_e
            dTadt = (self.G_ea * (T_e - T_a) - self.G_ab * (T_a - self.T_b) + self.P_Pa) / self.C_a

            dydt = np.array([dIdt, dTedt, dTadt])

        return dydt
    
    def solver(self):

        # Guess starting values (but still give some time to stabilize before t_eval begins)
        I_T0 = self.V_B / (self.R_S + 0.5 * self.R_T_max)
        T_e0 = self.T_b + (I_T0 **2 * 0.5 * self.R_T_max + self.heater.get_power(self.V_H) + self.P_Pe + (self.G_ea / (self.G_ea + self.G_ab)) * self.P_Pa) / (self.G_eb + (self.G_ea * self.G_ab) / (self.G_ea + self.G_ab))
        T_a0 = (self.G_ea * T_e0 + self.G_ab * self.T_b + self.P_Pa) / (self.G_ea + self.G_ab)

        y0 = np.array([I_T0, T_e0, T_a0])  # Starting values

        sol = sp.integrate.solve_ivp(
            fun=lambda t, y: self.rhs(t, y, self.n_comp),    # set ODEs
            t_span=(self.time[0]-1, self.time[-1]),         # evaluate over whole time window and a little before, so that the starting values can stabilize
            y0=y0,                                          # starting values
            method="RK45",                                  # "RK45"
            t_eval=self.time                               # points to give out (only for time values)
        )

        I_T, T_e, T_a = sol.y  # Shape: (3, N)
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
    ...
