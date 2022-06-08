import time
import numpy as np
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.signal_generator import make_signal

class Stimulation:
    def __init__(self, daq, duration, width=0, pulses=0, jitter=0, frequency=0, duty=0, width2=0, pulses2=0, jitter2=0, frequency2=0, duty2=0, pulse_type1='square', pulse_type2="square", name="", canal1=False, canal2=False):
        self.name = name
        self.daq = daq
        self.duration = duration
        self.exp = None

        self.type1 = pulse_type1
        self.pulses = pulses
        self.width = width
        self.duty = duty
        self.jitter = jitter
        self.freq = frequency

        self.type2 = pulse_type2
        self.pulses2 = pulses2
        self.width2 = width2
        self.duty2 = duty2
        self.jitter2 = jitter2
        self.freq2 = frequency2

    def __str__(self, indent=""):
        return_value = []
        if self.type1 == "random-square":
            return_value.append(indent+f"{self.name} - Canal 1 --- Duration: {self.duration}, Pulses: {self.pulses}, Width: {self.width}, Jitter: {self.jitter}")
        elif self.type1 == "square":
            return_value.append(indent+f"{self.name} -  Canal 1 --- Duration: {self.duration}, Frequency: {self.freq}, Duty: {self.duty}")
        if self.type2 == "random-square":
            return_value.append(indent+f"{self.name} - Canal 2 --- Duration: {self.duration}, Pulses: {self.pulses2}, Width: {self.width2}, Jitter: {self.jitter2}")
        elif self.type2 == "square":
            return_value.append(indent+f"{self.name} -  Canal 2 --- Duration: {self.duration}, Frequency: {self.freq2}, Duty: {self.duty2}")
        return "\n".join(return_value)
class Block:
    def __init__(self, name, data, delay=0, iterations=1, jitter=0):
        self.name = name
        self.data = data
        self.iterations = iterations
        self.delay = delay
        self.jitter = jitter
        self.exp = None

    def __str__(self, indent=""):
        stim_list = []
        for iteration in range(self.iterations):
            stim_list.append(indent + self.name + f" ({iteration+1}/{self.iterations}) --- Delay: {self.delay}, Jitter: {self.jitter}")
            for item in self.data:
                stim_list.append(item.__str__(indent=indent+"   "))
        return "\n".join(stim_list)

class Experiment:
    def __init__(self, blocks, framerate, exposition, mouse_id, directory, daq, name="No Name"):
        self.name = name
        self.blocks = blocks
        self.framerate = framerate
        self.exposition = exposition
        self.mouse_id = mouse_id
        self.directory = directory + f"/{name}"
        self.daq = daq

    def start(self, x_values, y_values):
        self.time, self.stim_signal = x_values, y_values
        self.daq.launch(self)

    def save(self, save, extents=None):
        if save is True:
            try:
                os.mkdir(self.directory)
            except Exception:
                pass
            with open(f'{self.directory}/experiment-metadata.txt', 'w') as file:
                file.write(f"Blocks\n{self.blocks.__str__()}\n\nFramerate\n{self.framerate}\n\nExposition\n{self.exposition}\n\nMouse ID\n{self.mouse_id}")
            
            dictionary = {
                "Blocks": self.blocks.__str__(),
                "Lights": self.daq.return_lights(),
                "Framerate": self.framerate,
                "Exposition": self.exposition,
                "Mouse ID": self.mouse_id
            }
            
            with open(f'{self.directory}/experiment-metadata.json', 'w') as file:
                json.dump(dictionary, file)
            
            self.daq.camera.save(self.directory, extents)
            self.daq.save(self.directory)
