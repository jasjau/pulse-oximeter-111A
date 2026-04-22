
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
from WF_SDK import device
from WF_SDK import scope
from WF_SDK import wavegen

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