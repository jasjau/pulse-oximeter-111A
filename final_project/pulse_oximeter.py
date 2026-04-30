
# Physics 111A Pulse Oximeter
# Jasmine Jaura and Melanie Linares

"""Physics 111A Lab 10 functions.
Written by Auden Young, 11/2025.

Important objects:
    ADSHardware (class): a collection of methods for interfacing with the ADS
        variables:
            handle: address to connect to the ADS
        functions:
            startup: connects to ADS
            open_scope: opens connection to oscilloscope
            trigger_scope: sets trigger level for scope (buggy)
            read_scope: collects data from oscilloscope
            close_scope: closes connection to oscilloscope
            use_wavegen: outputs function at wavegen
            close_wavegen: closes connection to wavegen
            disconnect: closes connection to ADS
    oscilloscope_run (function): opens connection to and collects data from scope
    fft (function): returns a fast fourier transform of input data
    demod_radio (function): demodulates a signal like we did for AM radio
    demod_lockin (function): does phase locked demodulation
    wavegen_functions (dict): easy names to access major types of functions wavegen can output
"""
import traceback
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd
from WF_SDK import device
from WF_SDK import scope
from WF_SDK import wavegen
from scipy.signal import find_peaks

# ADS FUNCTIONS # 
class ADSHardware():
    """Class of functions for interfacing with the ADS.
    """

    def __init__(self):
        self.handle = None

    def startup(self):
        """Connects to the ADS. Defines 'handle', the address to the ADS.
        Must be run at the beginning of every program using the ADS.
        """
        self.handle = device.open()

    def open_scope(self, buffer_size=1000, sample_freq=1e6):
        """Opens connection to the scope.

        Args:
            buffer_size (int, optional): How many data points are temporarily stored
            before being returned. The buffer is a temporary slot for storing a small amount of
            data before it is transferred to its final destination. Defaults to 1000.
            sample_freq (int, optional): How frequently the oscilloscope will sample
            from the input. Defaults to 1e6. You can decrease this if you have too
            many data points/the function is taking awhile to run for the time scale you need.
            (16e3 can be a reasonable selection.)
        """
        scope.open(self.handle, buffer_size=buffer_size, sampling_frequency=sample_freq)

    def trigger_scope(self, channel=1, level=0.1):
        """Sets trigger level for the scope. Kind of a buggy function; not used.

        Args:
            channel (int, optional): Selects which channel of scope to read out. 
            Defaults to 1.
            level (float, optional): Sets trigger level for scope. Defaults to 0.1.
        """
        scope.trigger(self.handle, enable=True, source=scope.trigger_source.analog, channel=channel,
                      edge_rising=True, level=level)

    def read_scope(self, channel=1):
        """Collects data from the scope.

        Args:
            channel (int, optional): Which channel to read from. Defaults to 1.

        Returns:
            buffer (array): An array of output data points. The buffer is a temporary slot 
            for storing a small amount of data before it is transferred to its final destination.
        """
        buffer = scope.record(self.handle, channel=channel)
        return buffer

    def close_scope(self):
        """Closes connection to the scope.
        """
        scope.close(self.handle)

    def use_wavegen(self, channel=1, function=wavegen.function.sine, offset_v=0, freq_hz=1e3, amp_v=1):
        """Runs the wavegen producing function with given parameters.

        Args:
            channel (int, optional): Which channel output is at. Defaults to 1.
            function (function object, optional): What type of function to output. 
            Defaults to wavegen.function.sine.
            offset (int, optional): Voltage offset (V). Defaults to 0.
            freq (int, optional): Frequency (Hz). Defaults to 1e3.
            amp (int, optional): Amplitude (V). Defaults to 1.
        """
        wavegen.generate(self.handle, channel=channel, function=function, offset=offset_v,
                         frequency=freq_hz, amplitude=amp_v)

    def close_wavegen(self):
        """Closes wavegen.
        """
        wavegen.close(self.handle)

    def disconnect(self):
        """Closes ADS connection. Must be run at the end of every program.
        """
        device.close(self.handle)

# OSCILLOSCOPE RUN # 
def oscilloscope_run(ads_object: ADSHardware, duration: int, channel: int, sampling_freq=500):
    """Collects data from the oscilloscope.

    Args:
        ads_object (ADSHardware object): the ADS being used
        duration (int): time length of trace to collect in seconds
        channel (int): which channel to collect data from
        sampling_freq (int, optional): How frequently the oscilloscope will sample
        from the input. Defaults to 1e6. You can decrease this if you have too
        many data points/the function is taking awhile to run for the time scale you need.
        (16e3 can be a reasonable selection.)

    Returns:
        data (dict): has two keys, "x" and "y" which have time (ms) and voltage (V) data
    """
    buffer_size = int(duration * sampling_freq) # number of samples to take
    data = {}
    ads_object.open_scope(sample_freq=sampling_freq, buffer_size=buffer_size)

    MS_CONVERSION = 1e3

    buffer = ads_object.read_scope()
    data["y"] = buffer

    # MODIFY THE LINE BELOW THIS ONE IN L10.2(d)
    data["x"] = np.arange(buffer_size) / (sampling_freq) * (MS_CONVERSION)

    ads_object.close_scope()
    return data

# FFT # 
def fft(data: dict):
    """Takes an FFT of input data.

    Args:
        data (dict): Provides x data in ms and y data in V obtained from oscilloscope.
    Returns:
        fft_result (dict): a dictionary with two keys, "frequencies" and "magnitudes",
                            containing the frequencies and magnitudes from the FFT.
    """
    fft_result = {}
    #FILL IN THIS FUNCTION FOR L10.3(b) and L10.3(c)
    MS_CONVERSION = 1e3
    #avg_timestep below may be helpful for your call to np.fft.fftfreq...
    avg_timestep = np.mean(np.diff(data["x"])/MS_CONVERSION)

    # consider np.fft.fft() and np.fft.fftfreq()
    voltage_data = data["y"]

    fft_result["frequencies"] = np.fft.fftfreq(len(voltage_data), avg_timestep)
    fft_result["magnitudes"] = (2/len(voltage_data)) * np.abs(np.fft.fft(voltage_data))

    are_geq_zero = []
    for f in fft_result["frequencies"]:
        if f < 0:
            are_geq_zero.append(False)
        else:
            are_geq_zero.append(True)

    positive_frequencies = []
    positive_frequencies_magnitudes = []
    i = 0
    for f in fft_result["frequencies"]:
        if are_geq_zero[i] == True:
            positive_frequencies.append(fft_result["frequencies"][i])
            positive_frequencies_magnitudes.append(fft_result["magnitudes"][i])
        i += 1

    fft_result["frequencies"] = positive_frequencies
    fft_result["magnitudes"] = positive_frequencies_magnitudes

    return fft_result

# LOW PASS FILTER # 
def butter_lowpass_filter(data, cutoff: float, fs: float, order=5):
    """Creates and applies a lowpass filter.

    Args:
        data (list): Provides y data in V obtained from oscilloscope.
        cutoff (float): 3 dB frequency (Hz) for low pass filter.
        fs (float): Sampling frequency data was taken at.
        order (int, optional): Order of the filter. Defaults to 5.

    Returns:
        list: Low pass filtered data in V.
    """
    # Define lowpass filter coefficients using butter function in scipy.signal package
    b, a = sig.butter(order, cutoff, btype='lowpass', analog=False, fs=fs, output='ba')
    # Applies lowpass filter using scipy.signal.filtfilt function
    y = sig.filtfilt(b, a, data)
    return y

# DEMOD RADIO # 
def demodulate_radio(data: dict, nu_3db: float, save=True):
    """Demodulate signal using the strategy we used for the AM radio.
    That is, first subtract the mean of the data, then do a lowpass filter.

    Args:
        data (dict): Provides x data in ms and y data in V obtained from oscilloscope.
        nu (float): 3 dB frequency (Hz) for low pass filter.
        save (bool, optional): Whether or not to save data to file. Defaults to True.

    Returns:
        demod_data (dict): has two keys, "x" and "y" which have time (ms) and voltage (V) data
    """
    demod_data = {}
    demod_data["x"] = data["x"]
    MILLISECOND_CONVERSION = 1e3

    #calculates average sampling frequency for digital filter
    fs = len(data["x"] - 1)*MILLISECOND_CONVERSION / (data["x"][-1] - data["x"][0])

    #FILL IN THESE LINES FOR L10.5(c)
    dc_offset_remove = np.array([(y-np.mean(data["y"])) for y in data["y"]]) #remove dc offset -- mean
    dc_offset_remove[dc_offset_remove < 0] = 0 #rectify negative parts are 0
    rectified_data = dc_offset_remove
    demod_data["y"] = butter_lowpass_filter(rectified_data, nu_3db, fs) #low pass

    #plot the different steps
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(demod_data["x"], data["y"])
    axs[0, 0].set_title('Raw Signal (Vout)')
    axs[0, 1].plot(demod_data["x"], dc_offset_remove, 'tab:orange')
    axs[0, 1].set_title('DC Offset Removed')
    axs[1, 0].plot(demod_data["x"], rectified_data, 'tab:green')
    axs[1, 0].set_title('Rectified (Vout1)')
    axs[1, 1].plot(demod_data["x"], demod_data["y"], 'tab:red')
    axs[1, 1].set_title('Low Pass Filtered (Vout2)')

    for ax in axs.flat:
        ax.set(xlabel='Time (ms)', ylabel='Voltage (V)')
        ax.grid(visible=True, which='major', color='black', linestyle='-')
        ax.grid(visible=True, which='minor', color='black', linestyle='--')
    
    for ax in axs.flat:
        ax.label_outer()
    
    plt.show()

    #save the data if desired
    if save:
        fname = os.path.join('./heartbeat_data', 'demod_lockin'+time.strftime("%Y%m%d-%H%M%S")+".txt")
        save_array = np.array([demod_data["x"], demod_data["y"]])
        np.savetxt(fname, save_array)

    return demod_data

# DEMOD LOCKIN # 
def demodulate_lockin(ads_object: ADSHardware, nu_mod: float, nu_3db: float, duration=5, channel=1, save=True):
    """Demodulate signal the way a lock in amplifier would, taking advantage
    of the fact that we can phase match.

    Args:
        ads_object (ADSHardware): the ADS being used.
        nu_mod (float): Modulation frequency (Hz). 100 recommended starting point.
        nu_3db (float): 3 dB frequency for low pass (Hz).
        duration (int, optional): Number of seconds to record for. Defaults to 5.
        channel (int, optional): Channel to read oscilloscope on. Defaults to 1.
        save (bool, optional): Whether or not to save data to file. Defaults to True.

    Returns:
        dict: _description_
    """
    MILLISECOND_CONVERSION = 1e3
    omega = 2*np.pi*nu_mod

    #we have to start the wavegen and oscilloscope read right after each other
    #in order to achieve phase locking
    test = ads_object.use_wavegen(channel=1, 
                    function=wavegen_functions["sine"], 
                    offset_v=2.1, 
                    freq_hz=nu_mod, 
                    amp_v=1)
    data = oscilloscope_run(ads_object, duration, 1, 500) #(ads_object, channel=channel, duration=duration)
    ads_object.close_wavegen()

    #calculates average sampling frequency for digital filter
    fs = len(data["x"] - 1)*MILLISECOND_CONVERSION / (data["x"][-1] - data["x"][0])

    demodulated_data = {}
    demodulated_data["x"] = data["x"]

    #calculate the cos and sin components of the local oscillator
    #i.e. what is being produced by wavegen
    demodulated_data["local_oscillator_cos"] = np.cos(omega*data["x"]/MILLISECOND_CONVERSION)
    demodulated_data["local_oscillator_sin"] = np.sin(omega*data["x"]/MILLISECOND_CONVERSION)

    #FILL IN THE BLANKS BELOW FOR L10.6(a)
    #finds the cos and sin components of the signal read on the scope
    demodulated_data["sin"] = data["y"] * demodulated_data["local_oscillator_sin"]
    demodulated_data["cos"] = data["y"] * demodulated_data["local_oscillator_cos"]

    #low pass filters the data
    demodulated_data["lowpass_sin"] = butter_lowpass_filter(demodulated_data["sin"], nu_3db, fs)
    demodulated_data["lowpass_cos"] = butter_lowpass_filter(demodulated_data["cos"], nu_3db, fs)

    #adds sin and cos components in quadrature to obtain the demodulated signal
    demodulated_data["y"] = np.sqrt(demodulated_data["lowpass_cos"]**2 + demodulated_data["lowpass_sin"]**2)

    #plot the steps to get demodulated signal
    '''fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(demodulated_data["x"], data["y"])
    axs[0, 0].set_title('Raw Signal')
    axs[0, 1].plot(demodulated_data["x"], demodulated_data["sin"], 'tab:orange')
    axs[0, 1].plot(demodulated_data["x"], demodulated_data["cos"], 'tab:green')
    axs[0, 1].set_title('Sin & Cos components')
    axs[1, 0].plot(demodulated_data["x"], demodulated_data["local_oscillator_cos"])
    axs[1, 0].plot(demodulated_data["x"], demodulated_data["local_oscillator_sin"])
    axs[1, 0].set_title('Local oscillator')
    axs[1, 1].plot(demodulated_data["x"], demodulated_data["lowpass_cos"])
    axs[1, 1].plot(demodulated_data["x"], demodulated_data["lowpass_sin"])
    axs[1, 1].set_title("Filtered sin & cos components")'''

    '''for ax in axs.flat:
        ax.set(xlabel='Time (ms)', ylabel='Voltage (V)')
        ax.grid(visible=True, which='major', color='black', linestyle='-')
        ax.grid(visible=True, which='minor', color='black', linestyle='--')
    
    for ax in axs.flat:
        ax.label_outer()
    
    plt.show()

    #plot the final demodulated signal
    plt.plot(demodulated_data["x"], demodulated_data["y"])
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (V)")
    plt.show()'''

    #save the data if desired
    '''if save:
        fname = os.path.join('./heartbeat_data', 'demod_lockin'+time.strftime("%Y%m%d-%H%M%S")+".txt")
        save_array = np.array([demodulated_data["x"], demodulated_data["y"]])
        np.savetxt(fname, save_array)'''

    return demodulated_data

wavegen_functions = {"sine":wavegen.function.sine, "square":wavegen.function.square,
                     "triangle":wavegen.function.triangle, "dc":wavegen.function.dc}

def demodulate_lockin_from_data(data, nu_mod: float, nu_3db: float):

    MILLISECOND_CONVERSION = 1e3
    omega = 2 * np.pi * nu_mod

    demodulated_data = {}
    demodulated_data["x"] = data["x"]

    demodulated_data["local_oscillator_cos"] = np.cos(omega * data["x"] / MILLISECOND_CONVERSION)
    demodulated_data["local_oscillator_sin"] = np.sin(omega * data["x"] / MILLISECOND_CONVERSION)

    demodulated_data["sin"] = data["y"] * demodulated_data["local_oscillator_sin"]
    demodulated_data["cos"] = data["y"] * demodulated_data["local_oscillator_cos"]

    fs = len(data["x"] - 1)*MILLISECOND_CONVERSION / (data["x"][-1] - data["x"][0])

    demodulated_data["lowpass_sin"] = butter_lowpass_filter(demodulated_data["sin"], nu_3db, fs)
    demodulated_data["lowpass_cos"] = butter_lowpass_filter(demodulated_data["cos"], nu_3db, fs)

    demodulated_data["y"] = np.sqrt(demodulated_data["lowpass_cos"]**2 + demodulated_data["lowpass_sin"]**2)

    return demodulated_data

if __name__ == "__main__":
    ads = ADSHardware()
    ads.startup()


    try:
        '''# 1 run of the LED collecteing data
        # ads.use_wavegen(channel=1, function=wavegen_functions["dc"], offset_v=2) #ADS control
    
        # time.sleep(1)

        # duration = 10
        # raw_data = oscilloscope_run(ads, duration, 1, 500) # scope control

        # plt.plot(raw_data["x"], raw_data["y"])
        # plt.xlabel('Time (ms)')
        # plt.ylabel('Voltage (V)')
        # plt.title("Scope Trace (Raw Data)")
        # plt.show()
        # ads.close_wavegen()
        ads.use_wavegen(channel=1, function=wavegen_functions["square"], offset_v=2, amp_v=2, freq_hz=7000)
        ads.use_wavegen(channel=2, function=wavegen_functions["square"], offset_v=2, amp_v=2, freq_hz=12000)

        time.sleep(1)
       
        duration = 30
        sampling_freq = 5000
        raw_data = oscilloscope_run(ads, duration=duration, channel=1, sampling_freq=sampling_freq)
       
        # PLOT RAW DATA
        plt.plot(raw_data["x"], raw_data["y"])
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (V)')
        plt.title("Scope Trace (Raw Data)")
        plt.show()
        ads.close_wavegen()

        # TAKE FFT
        fft_raw = fft(raw_data)
        plt.plot(fft_raw["frequencies"], fft_raw["magnitudes"])
        plt.xlabel("Frequencies (Hz)")
        plt.ylabel("Voltage (V)")
        plt.grid(visible=True, which='major', color='black', linestyle='-')
        plt.grid(visible=True, which='minor', color='black', linestyle='--')
        #plt.ylim(0, 0.1)
        plt.title("FFT Demod Radio")
        plt.show()
        #plt.xlim(0, 5)
        
        # demod_radio_raw = demodulate_radio(raw_data, nu_3db=15000)

        # Run again to take demodulated lock in
        red_demodlockin = demodulate_lockin(ads, nu_mod=7000, nu_3db=100)
        ir_demodlockin = demodulate_lockin(ads, nu_mod=12000, nu_3db=100)

        # Calculate
        # red
        red_peaks, _ = find_peaks(red_demodlockin["y"]) # gets indices
        red_troughs, _ = find_peaks(-red_demodlockin["y"]) # gets indices
        red_vpp = np.mean(red_demodlockin["y"][red_peaks]) - np.mean(red_demodlockin["y"][red_troughs])

        # ir
        ir_peaks, _ = find_peaks(ir_demodlockin["y"]) # gets indices
        ir_troughs, _ = find_peaks(-ir_demodlockin["y"]) # gets indices
        ir_vpp = np.mean(ir_demodlockin["y"][ir_peaks]) - np.mean(ir_demodlockin["y"][ir_troughs])

        red_AC_rms = red_vpp / 2
        ir_AC_rms = ir_vpp / 2

        # DC needs to be moving average of demodlockin data
        window = int(0.5 * sampling_freq)
        red_series = pd.Series(red_demodlockin["y"])
        ir_series = pd.Series(ir_demodlockin["y"])
        red_DC = red_series.rolling(window, center=True).mean()
        ir_DC = ir_series.rolling(window, center=True).mean()
        # red_DC = np.mean(red_demodlockin["y"])
        # ir_DC = np.mean(ir_demodlockin["y"])

        ratio = (red_AC_rms/red_DC) / (ir_AC_rms/ir_DC)

        fig, ax = plt.subplots(2, 2)
        #ax[0,0].plot(raw_data["x"], raw_data["y"]); ax[0,0].set_title("Scope Trace (Raw Data)")
        #ax[0,1].plot(red_demodlockin["x"], ratio); ax[0,1].set_title("Time v Ratio")
        #ax[1,0].plot(x, y3); ax[1,0].set_title("Tangent")
        # # ax[1,1].plot(x, y4); ax[1,1].set_title("Tanh")
        # plt.show()

        print(f"red ac rms: {red_AC_rms}")
        print(f"red_dc: {red_DC}")
        print(f"ir rms: {ir_AC_rms}")
        print(f"ir_dc: {ir_DC}")
        print(f"ratio: {ratio}")
        print(f"AC same? {red_AC_rms == ir_AC_rms}")
        print(f"DC same? {red_DC == ir_DC}")
        #plt.plot(red_demodlockin["x"], ratio)
        #plt.xlabel('Time (ms)')
        #plt.ylabel('R')
        #plt.show()


        ads.disconnect()'''

        # Set-up
        red_freq = 150
        ir_freq  = 500

        ads.use_wavegen(channel=1, function=wavegen_functions["square"], offset_v=2, amp_v=2, freq_hz=red_freq)
        ads.use_wavegen(channel=2, function=wavegen_functions["square"], offset_v=2, amp_v=2, freq_hz=ir_freq)

        time.sleep(1)

        # Take raw data
        duration = 8
        sampling_freq = 1100

        raw_data = oscilloscope_run(ads, duration=duration, channel=1, sampling_freq=sampling_freq)

        ads.close_wavegen()

        # Save raw data
        # fname = os.path.join('./saved_files', 'raw_'+time.strftime("%Y%m%d-%H%M%S")+".txt")
        # np.savetxt(fname, np.array([raw_data["x"], raw_data["y"]]))

        # Plot Raw Data
        fig1, ax1 = plt.subplots()
        ax1.set_title("Scope Trace (Raw Data)")
        ax1.plot(raw_data["x"], raw_data["y"])
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True)

        # Take FFT and Plot
        fft_raw = fft(raw_data)
        fig2, ax2 = plt.subplots()
        ax2.set_title("FFT of Raw Data")
        ax2.plot(fft_raw["frequencies"], fft_raw["magnitudes"])
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude (V)")
        ax2.grid(True)

        # Lock-in Demodulation
        red = demodulate_lockin_from_data(data=raw_data, nu_mod=red_freq, nu_3db=5)
        ir  = demodulate_lockin_from_data(data=raw_data, nu_mod=ir_freq, nu_3db=5)

        # Save Demod Data
        # fname = os.path.join('./saved_files', 'demod_'+time.strftime("%Y%m%d-%H%M%S")+".txt")
        # np.savetxt(fname, np.array([red["x"], red["y"], ir["y"]]))

        # Get AC and DC
        window = int(0.5 * sampling_freq)

        red_series = pd.Series(red["y"])
        ir_series  = pd.Series(ir["y"])

        red_DC = red_series.rolling(window, center=True).mean() # dc
        ir_DC  = ir_series.rolling(window, center=True).mean() # dc

        red_AC = red_series - red_DC # remove dc from ac signal
        ir_AC  = ir_series - ir_DC # remove dc from ac signal

        # CHECK THESE CALCULATIONS
        red_peaks, _ = find_peaks(red["y"]) # gets indices
        red_troughs, _ = find_peaks(-red["y"]) # gets indices
        red_vpp = np.mean(red["y"][red_peaks]) - np.mean(red["y"][red_troughs])

        # ir
        ir_peaks, _ = find_peaks(ir["y"]) # gets indices
        ir_troughs, _ = find_peaks(-ir["y"]) # gets indices
        ir_vpp = np.mean(ir["y"][ir_peaks]) - np.mean(ir["y"][ir_troughs])

        red_AC_rms = red_vpp / 2
        ir_AC_rms = ir_vpp / 2
        # red_AC_rms = red_AC.rolling(window, center=True).std() # ac rms-- rolling standard deviation
        # ir_AC_rms  = ir_AC.rolling(window, center=True).std() # ac rms-- rolling standard deviation

        # ratio
        ratio = (red_AC_rms / red_DC) / (ir_AC_rms / ir_DC) # ratio as a function of time

        avg_ratio = np.mean(ratio)
        print(f"Average ratio: {np.mean(ratio)}")

        # plot
        fig3, ax3 = plt.subplots()
        ax3.set_title("Demodulated Signals (Offset of +0.01V on Red Signal)")
        ax3.plot(red["x"], red["y"] + 0.01, label="Red", color="red", linewidth=1)
        ax3.plot(ir["x"], ir["y"], label="IR", color="purple", linewidth=1)
        ax3.legend()
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Voltage")
        ax3.grid(True)

        fig4, ax4 = plt.subplots()
        ax4.set_title("Ratio (R/IR) vs Time (ms)")
        ax4.plot(red["x"], ratio)
        ax4.set_xlabel("Time (ms)")
        ax4.set_ylabel("R")
        ax4.grid(True)


        # Plot SpO2 vs Ratio
        A = 113.33
        B = 26.67
        spo2_val = A - (B * avg_ratio)
        R_range = np.linspace(0.4, 2.2, 100)
        SpO2_line = A - B * R_range

        fig5, ax5 = plt.subplots(figsize=(9, 6))
        ax5.plot(R_range, SpO2_line, 'b--', alpha=0.6, label=f'Calibration: $SpO_2 = {A} - {B}R$')
        ax5.scatter([0.5, 2.0], [100, 60], color='black', marker='x', label='Reference Points')
        ax5.scatter(avg_ratio, spo2_val, color='pink', s=150, edgecolors='black', zorder=5, 
            label=f'Your Measurement: {spo2_val:.1f}%')
        ax5.text(avg_ratio + 0.05, spo2_val + 2, f"You: {spo2_val}%", color='red', fontweight='bold')
        ax5.set_title("SpO2 vs. Ratio (R/IR)")
        ax5.set_xlabel("R/IR")
        ax5.set_ylabel("SpO2 (%)")
        ax5.set_ylim(50, 115)
        ax5.legend()

        
        plt.show()

        ads.disconnect()

       
    except Exception:
        #allows you to see errors while ensuring that connections closed
        traceback.print_exc()
        ads.close_scope()
        ads.close_wavegen()
        ads.disconnect()