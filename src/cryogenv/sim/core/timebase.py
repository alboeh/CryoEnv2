import numpy as np

def init_timebase(self, record_length, sample_frequency):
    self.record_length = record_length
    self.sample_frequency = sample_frequency
    self.time = np.arange(record_length) / sample_frequency     # time vector
    self.f = np.fft.rfftfreq(record_length, 1/sample_frequency)  # Frequency vector
    self.w = 2*np.pi*self.f         # angular frequency vector
    return self.time, self.f, self.w