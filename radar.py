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
    return 10 ** (db / 10)
  
def volt_to_dbm(v_peak):
    return w_to_dbm((v_peak / math.sqrt(2)) ** 2 / 50)
    
def noise_power(temp, bandwidth):
    boltz = 1.38 * 10 ** (-23)
    return w_to_dbm(boltz * temp * bandwidth)
    
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
        result = dbm_to_w(self.transmit_power) * db_to_linear(self.antenna_gain) ** 2 * wavelength ** 2 * \
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
    
    def get_max_rx_power(self):
        return volt_to_dbm(self.get_adc_max_input())
 
    def get_min_rx_power(self):
        return volt_to_dbm(self.get_adc_min_input())
 
    def get_noise_power(self, kelvin):
        return w_to_dbm(BOLTZMAN_CONSTANT * kelvin * self.bandwidth)        
    
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
        
    def set_receive_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        
    def set_target_cross_section(self, cross_section):
        self.target_cross_section = cross_section

radar = radar_model()

radar.set_transmit_power(30)
radar.set_antenna_gain(20)
radar.set_frequency(440e6)

print(radar.get_range_estimate(radar.get_min_rx_power()))


"""
        
adc_bits = 12
adc_fullscale = 2
adc_resolution = adc_fullscale / (2**adc_bits)
adc_gain = 1000

# Peak input voltages
adc_max_input = adc_fullscale / adc_gain
adc_min_input = (adc_resolution * 0.5) / adc_gain
adc_dynamic_range = 20 * math.log10(adc_max_input / adc_min_input)

max_rx_power = volt_to_dbm(adc_max_input)
min_rx_power = volt_to_dbm(adc_min_input)

tx_power = 30

max_range = radar_range(min_rx_power, tx_power, 0.75, 10, 20)
min_range = radar_range(max_rx_power, tx_power, 0.75, 10, 20)
    
print("Gain = ", adc_gain, " Min range: ", min_range, " Max Range :", max_range, " DR = ", adc_dynamic_range, " Ratio = ", max_range / min_range)

num_points = 1000

xpoints = np.empty(num_points)
ypoints = np.empty(num_points)
ypoints2 = np.empty(num_points)

step = adc_gain / num_points

for i in range(0, num_points):
    gain = step * (i + 1)

    adc_max_input = adc_fullscale / gain
    adc_min_input = (adc_resolution * 0.5) / gain
    adc_dynamic_range = 20 * math.log10(adc_max_input / adc_min_input)

    max_rx_power = volt_to_dbm(adc_max_input)
    min_rx_power = volt_to_dbm(adc_min_input)

    tx_power = 17

    max_range = radar_range(min_rx_power, tx_power, 0.125, 10, 10)
    min_range = radar_range(max_rx_power, tx_power, 0.125, 10, 10)

    xpoints[i] = 10 * math.log10(gain)
    ypoints[i] = min_range
    ypoints2[i] = max_range    

plt.title("Minimum and maximum radar range as a function of IF gain.\n 12 bit ADC, full-scale = 2V, Ptx = 17 dBm")
plt.xlabel("Gain (dB)")
plt.ylabel("Minimum range (m)")
plt.plot(xpoints, ypoints)
plt.twinx()
plt.ylabel("Maximum range (m)")
plt.plot(xpoints, ypoints2)
plt.show()
"""