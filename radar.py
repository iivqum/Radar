import math
import matplotlib.pyplot as plt
import numpy as np

BOLTZMAN_CONSTANT = 1.38 * 10**(-23)
# meters per second
WAVE_VELOCITY = 3 * 10 ** 8

def dbm_to_w(dbm):
    return 10 ** (dbm / 10) * 0.001
  
def w_to_dbm(w):
    return 10 * math.log10(w * 1000)
    
def dbm_to_volt(dbm):
    return math.sqrt(dbm_to_w(dbm) * 50)
  
def db_to_linear(db):
    return 10 ** (db / 20)
  
def volt_to_dbm(v_peak):
    return w_to_dbm((v_peak / math.sqrt(2)) ** 2 / 50)
    
def rms_to_peak(rms):
    return rms * math.sqrt(2)
    
def peak_to_rms(peak):
    return peak / math.sqrt(2)

class radar_model:
    def __init__(self):
        # Hz
        self.frequency = 400e6
        self.bandwidth = 1e6
        # dBm
        self.transmit_power = 30
        # dBi, assume RX and TX antennas are identical
        self.antenna_gain = 20
        self.adc_bits = 12
        # Maximum ADC voltage, volts
        self.adc_fullscale = 2
        # dB
        self.adc_gain = 20
        # square meters
        self.target_cross_section = 10
        
    def get_range_estimate(self, rx_power_dbm):
        if rx_power_dbm > self.transmit_power:
            print("Receive power is greater than transmit power!")
            
        wavelength = WAVE_VELOCITY / self.frequency
        result = dbm_to_w(self.transmit_power) * (10 ** (self.antenna_gain / 10)) ** 2 * wavelength ** 2 * \
            self.target_cross_section / ((4 * math.pi)**3 * dbm_to_w(rx_power_dbm))
            
        return result ** (1/4)        
        
    def get_adc_resolution(self):
        return self.adc_fullscale / (2 ** self.adc_bits)
    
    def get_adc_min_input(self):
        return self.get_adc_resolution() * 0.5 / db_to_linear(self.adc_gain)
    
    def get_adc_max_input(self):
        return self.adc_fullscale / db_to_linear(self.adc_gain)
    
    def get_adc_dynamic_range(self):
        return 20 * math.log10(self.adc_fullscale / self.get_adc_resolution() * 0.5)
    
    def get_enob(self):
        noise_voltage = dbm_to_volt(self.get_noise_power(300)) * math.sqrt(2)
    
        return noise_voltage / (self.get_adc_resolution() * 0.5)
    
    def get_max_rx_power(self):
        return volt_to_dbm(self.get_adc_max_input())
 
    def get_min_rx_power(self):
        return volt_to_dbm(self.get_adc_min_input())
 
    def get_noise_power(self, kelvin):
        return w_to_dbm(BOLTZMAN_CONSTANT * kelvin * self.bandwidth)       

    def get_wavelength(self):
        return WAVE_VELOCITY / self.frequency
    
    def set_frequency(self, frequency):
        self.frequency = frequency
        
    def set_adc_bits(self, bits):
        self.adc_bits = abs(bits)
        
    def set_adc_gain(self, gain):
        self.adc_gain = gain
        
    def set_adc_fullscale(self, fullscale):
        self.adc_fullscale = fullscale
        
    def set_transmit_power(self, power):
        self.transmit_power = power

    def set_antenna_gain(self, gain):
        self.antenna_gain = gain
        
    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        
    def set_target_cross_section(self, cross_section):
        self.target_cross_section = cross_section
        
class radar_model_fmcw(radar_model):
    def __init__(self):
        super().__init__()
        
        self.chirp_frequency = 1000
        self.chirps = 10
        
    def get_if(self, max_range):
        return self.chirp_frequency * 2 * max_range * self.bandwidth / (WAVE_VELOCITY)

    def get_range_resolution(self):
        return WAVE_VELOCITY / (2 * self.bandwidth)
    
    def get_velocity_resolution(self):
        return self.get_wavelength() * self.chirp_frequency / (2 * self.chirps)
    
    def get_max_velocity(self):
        return self.get_wavelength() * 0.25 * self.chirp_frequency
    
    def get_max_range_fixed_if(self, max_if):
        return max_if / (self.chirp_frequency * 2 * self.bandwidth) * WAVE_VELOCITY
    
    def set_chirp_frequency(self, chirp_frequency):
        self.chirp_frequency = chirp_frequency
        
    def set_chirps(self, chirps):
        self.chirps = chirps
        
    def print_parameters(self, max_if):
        range_res = radar.get_range_resolution()
        range_max = radar.get_max_range_fixed_if(max_if)
        vel_max = radar.get_max_velocity()
        vel_res = radar.get_velocity_resolution()
        
        print("Max range : ", range_max)
        print("Max velocity : ", vel_max)       
        print("Range resolution : ", range_res)
        print("Velocity resolution : ", vel_res)
        print("Range error : ", range_res / range_max * 100)
        print("Velocity error : ", vel_res / vel_max * 100)

radar = radar_model_fmcw()
radar.set_transmit_power(47)
radar.set_antenna_gain(20)
radar.set_frequency(2.4e9)
radar.set_bandwidth(100e6)
radar.set_chirp_frequency(100)
radar.set_chirps(100)

radar.print_parameters(20e3)

"""
radar = radar_model()

radar.set_transmit_power(47)
radar.set_antenna_gain(20)
radar.set_frequency(440e6)
radar.set_adc_bits(8)
radar.set_adc_fullscale(2)
radar.set_bandwidth(10e6)

print(radar.get_enob())

num_points = 1000
xpoints = np.empty(num_points)
ypoints = np.empty(num_points)
ypoints2 = np.empty(num_points)

step = 40 / num_points

for i in range(0, num_points):
    gain = step * (i + 1)

    radar.set_adc_gain(gain)
    
    min_range = radar.get_range_estimate(radar.get_max_rx_power())
    max_range = radar.get_range_estimate(radar.get_min_rx_power())
    
    xpoints[i] = gain
    ypoints[i] = min_range
    ypoints2[i] = max_range
    
plt.title("Minimum and maximum radar range as a function of (noiseless) IF gain.\n 12 bit ADC, full-scale = 2V, Ptx = 30 dBm.")
plt.xlabel("Gain (dB)")
plt.ylabel("Minimum range (m)")
plt.plot(xpoints, ypoints)
plt.twinx()
plt.ylabel("Maximum range (m)")
plt.plot(xpoints, ypoints2)
plt.show()
"""