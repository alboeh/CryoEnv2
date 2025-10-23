import numpy as np

from ..core.timebase import init_timebase
from ..core.constants import const

class NOISE:
    """
    ...
    "simple": 
    Means I just fittet the noise to real data.
    It contains a constant part (from SQUID and rest), which decreases through transition.
    and the 1/f noise part, which in biggest during the transition, before and after nearly 0.
    Both are lowpass filtered in the end.
    
    
    """
    # central place for defaults
    DEFAULTS = {"S_squid": 1.44e-16,    # SQUID noise
                "S_const": 1e-13,       # Constant noise (Johnson TES, Johnson Shunt, Thermal)
                "mu_flicker": 2.1e-10,  # Fit parameter for flicker noise
                "ceta_flicker": 2,     # Fit parameter for flicker noise
                "f_cut": 10e3           # Cutoff frequency in Line hardware filter
                }
    
    AVAILABLE_MODELS = ["simple"]

    def __init__(self, *, record_length, sample_frequency, default: bool = False):
        """
        ...
        """
        self.sample_frequency = sample_frequency
        self.record_length = record_length
        self.time, self.f, self.w = init_timebase(self, record_length, sample_frequency)
        self.N = len(self.w)

        if default:
            self.S_squid = self.DEFAULTS["S_squid"]
            self.S_const = self.DEFAULTS["S_const"]
            self.mu_flicker = self.DEFAULTS["mu_flicker"]
            self.ceta_flicker = self.DEFAULTS["ceta_flicker"]
            self.f_cut = self.DEFAULTS["f_cut"]
        else:
            raise ValueError("If default=False, you must provide ...")
            # TODO: Implement


        print(f"Noise applied.")

    def get_total_noise(self, I_T, T_e, squid, tes, model="simple"):
        fs = self.sample_frequency
        N = self.record_length

        if model == "simple":
            self.S_flicker = self.mu_flicker * ((2 * np.pi) / np.maximum(self.w, self.w[1]))**self.ceta_flicker     # Preventing div/0
        else:
            raise ValueError("Only simple noise model yet implmented")
        
        # In this part we want to adapt the noise depending on the current position in the Transition
        # As far as we measured in real detectors, the 1/f flicker noise only happens during transition, not before and only a little after
        # The constant noise is increasing from normal to superconducting, which is weird but observed in different setups
        R_T = tes.get_resistance(T_e[0])
        ratio = R_T / tes.R_max           # 0 if superconducting, ~mid if in transition, 1 if normal conducting

        # Adapt noises as explained
        S_const_ratio = self.S_const * (1 + ratio)    # Measured in real detector, that the const noise is ~doubling from super to normal conducting
        S_flicker_ratio = self.S_flicker * np.sin(np.pi * ratio)
        S_squid_ratio = self.S_squid    # Change nothing here
        
        # Sum up all noise contributions
        S_ratio = S_const_ratio + S_squid_ratio + S_flicker_ratio

        # Apply the Lowpass filtering
        LP = np.abs(3 / ((1j*self.w / (2 * np.pi * self.f_cut))**2 + 3*(1j*self.w/(2*np.pi*self.f_cut))+3)) **2
        S_LP = S_ratio * LP 
        S_const_LP = S_const_ratio * LP
        S_flicker_LP = S_flicker_ratio * LP
        S_squid_LP = S_squid_ratio * LP

        phase_rnd = np.random.uniform(0, 2 * np.pi, self.N) # random phase for noise
        
        df = fs / N
        noise_rnd = np.sqrt(S_LP * df / 2) * (np.cos(phase_rnd) + 1j * np.sin(phase_rnd))

        # These are just for nice plotting:
        noise_rnd_const = np.sqrt(S_const_LP * df / 2) * (np.cos(phase_rnd) + 1j * np.sin(phase_rnd))
        noise_rnd_flicker = np.sqrt(S_flicker_LP * df / 2) * (np.cos(phase_rnd) + 1j * np.sin(phase_rnd))
        noise_rnd_squid = np.sqrt(S_squid_LP * df / 2) * (np.cos(phase_rnd) + 1j * np.sin(phase_rnd))

        # TODO: Why do I need such a big factor here? Should actually work without
        #noise_rnd *= 1e0

        # Noise gets added to Current Bias
        self.I_T_noise = I_T + np.fft.irfft(noise_rnd,  n=N)

        # Apply SQUID amplification (only relevant for NPS plotting scaling)
        # This is necessary, because noise was calculated for I_T, but the "real" noise we only see after the SQUID stage amplification
        self.NPS = (2 / (fs * N)) * np.abs(np.fft.rfft(squid.get_output(np.fft.irfft(noise_rnd))))**2  # [A²/Hz]
        self.NPS_const = (2 / (fs * N)) * np.abs(np.fft.rfft(squid.get_output(np.fft.irfft(noise_rnd_const))))**2  # [A²/Hz]
        self.NPS_flicker = (2 / (fs * N)) * np.abs(np.fft.rfft(squid.get_output(np.fft.irfft(noise_rnd_flicker))))**2  # [A²/Hz]
        self.NPS_squid = (2 / (fs * N)) * np.abs(np.fft.rfft(squid.get_output(np.fft.irfft(noise_rnd_squid))))**2  # [A²/Hz]