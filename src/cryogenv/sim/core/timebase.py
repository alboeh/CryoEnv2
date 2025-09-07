import numpy as np

def init_timebase(self, record_length, sample_frequency):
    self.record_length = record_length
    self.sample_frequency = sample_frequency
    self.time = np.arange(record_length) / sample_frequency     # time vector