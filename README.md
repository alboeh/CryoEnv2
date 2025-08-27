# CryoEnv2
This is a new Varsion of the CryoEnv by Felix Wagner, but adapted to COSINUS detectors as well as new models.

For data handling we are using the well-known cait package from Felix Wagner, because it has some great features which can be useful lateron. Additionally it makes sure, that the Testpulses have the same format in the end, so we make it compatible with the "real" traces.

Currently open topics:
- Implement 3-component (COSINUS) model by V. Zema
- Implement SQUID
- Implement noise as well as SNR output
- Implement load resistor of the Bias voltage source
- Get TES curves randomized
- Get TES dependent on I_B
- Integrate in gymnasium environment wrapper
- How to handle Pile-Ups, saturation or instable OPs (bumps in the transition)?