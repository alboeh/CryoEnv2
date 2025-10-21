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

    def get_total_noise(self, I_T, squid, model="simple"):
        fs = self.sample_frequency
        N = self.record_length

        if model == "simple":
            self.S_flicker = self.mu_flicker * ((2 * np.pi) / np.maximum(self.w, self.w[1]))**self.ceta_flicker     # Preventing div/0
        else:
            raise ValueError("Only simple noise model yet implmented")
        
        # Sum up all noise contributions
        self.S = self.S_const + self.S_squid + self.S_flicker

        # Apply the Lowpass filtering
        LP = np.abs(3 / ((1j*self.w / (2 * np.pi * self.f_cut))**2 + 3*(1j*self.w/(2*np.pi*self.f_cut))+3)) **2
        self.S *= LP 
        self.S_const *= LP
        self.S_flicker *= LP
        self.S_squid *= LP

        phase_rnd = np.random.uniform(0, 2 * np.pi, self.N) # random phase for noise
        
        df = fs / N
        noise_rnd = np.sqrt(self.S * df / 2) * (np.cos(phase_rnd) + 1j * np.sin(phase_rnd))

        # These are just for nice plotting:
        noise_rnd_const = np.sqrt(self.S_const * df / 2) * (np.cos(phase_rnd) + 1j * np.sin(phase_rnd))
        noise_rnd_flicker = np.sqrt(self.S_flicker * df / 2) * (np.cos(phase_rnd) + 1j * np.sin(phase_rnd))
        noise_rnd_squid = np.sqrt(self.S_squid * df / 2) * (np.cos(phase_rnd) + 1j * np.sin(phase_rnd))

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