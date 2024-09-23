import json
import os
import time
import serial
from serial.tools.list_ports import comports
import matplotlib.pyplot as plt
import numpy as np




def test_acquisition():
    print("Stop this test by closing the graph and type KeyboardInterrupt (ctrl-C)...")
    fig, ax = plt.subplots(1)
    line_led0, = ax.plot([], [], label="Isobestic", c="m")
    line_led1, = ax.plot([], [], label="Fluorescence", c="b")

    led0_signal = np.zeros(100) # 405 nm
    led1_signal = np.zeros(100) # 470 nm
    plt.ylim(10000, 1e12)
    plt.xlim(0, 100)
    plt.xticks([0, 20, 40, 60, 80, 100], range(6))
    plt.xlabel("Time [s]")
    plt.ylabel("Intensity [a.u.]")
    plt.legend()
    plt.pause(0.01)

    arduino.reset_input_buffer()
    arduino.write(b"3")

    try:
        while True:
            data = (arduino.readline()).decode('utf-8')
            try:
                if int(data[0]) == 1:
                    led0_signal = np.roll(led0_signal, -1)
                    led0_signal[1] = int(data[-12:])

                if int(data[1]) == 1:
                    led1_signal = np.roll(led1_signal, -1)
                    led1_signal[1] = int(data[-12:])

            except BaseException:
                pass

            line_led0.set_data(range(100), led0_signal)
            line_led1.set_data(range(100), led1_signal)
            plt.pause(0.0001)
    except KeyboardInterrupt:
        arduino.write(b"0")
        print("- - - Ending Fused Fiber Photometry Software - - -")


def timed_acquisition():
    print("\n\nImportant informations")
    saving_path = input("Path where to save the data: ")
    experiment_length = input("Experiment length in seconds: ")
    mouse_id = input("Mouse ID: ")
    other_info = input("Other info: ")

    metadata = {"experiment_length":experiment_length,
                "mouse_id": mouse_id,
                "other_info": other_info}

    date = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

    with open(os.path.join(saving_path, f"metadata_{mouse_id}_{date}.json"),
              "w", encoding="utf-8") as outfile:
        json.dump(metadata, outfile)

    print("\n\n- - Starting acquisition - - ")

    with open(os.path.join(saving_path, f"data_{mouse_id}_{date}.txt"), "w",
              encoding="utf-8") as file:
        file.write("time\tarduino_output\n")

        fig, ax = plt.subplots(1)
        line_led0, = ax.plot([], [], label="Isobestic", c="m")
        line_led1, = ax.plot([], [], label="Fluorescence", c="b")

        led0_signal = np.zeros(100) # 405 nm
        led1_signal = np.zeros(100) # 470 nm
        plt.ylim(10000, 1e12)
        plt.xlim(0, 100)
        plt.xticks([0, 20, 40, 60, 80, 100], range(6))
        plt.xlabel("Time [s]")
        plt.ylabel("Intensity [a.u.]")
        plt.legend()
        plt.pause(0.01)

        arduino.reset_input_buffer()
        arduino.write(b"3")

        limit_time = time.perf_counter() + int(experiment_length) + 2.1 # Dont ask why, just do it.

        print(f"End: {time.strftime('%H:%M:%S', time.localtime(limit_time))}")

        while time.perf_counter() <= limit_time:
            data = (arduino.readline()).decode('utf-8')

            try:
                if int(data[0]) == 1:
                    led0_signal = np.roll(led0_signal, -1)
                    led0_signal[1] = int(data[-12:])

                if int(data[1]) == 1:
                    led1_signal = np.roll(led1_signal, -1)
                    led1_signal[1] = int(data[-12:])

            except BaseException:
                pass


            line_led0.set_data(range(100), led0_signal)
            line_led1.set_data(range(100), led1_signal)

            file.write(f"{time.perf_counter_ns()}\t{data}")
            plt.pause(0.0001)

    arduino.write(b"0")


    print("- - Acquisition Over - -\n\n")
    print("- - - Ending Fused Fiber Photometry Software - - -")


def unlimited_acquisition():
    
    print("\n\nImportant informations")
    saving_path = input("Path where to save the data: ")
    mouse_id = input("Mouse ID: ")
    other_info = input("Other info: ")


    metadata = {"experiment_length":"unlimited acquisition",
                "mouse_id": mouse_id,
                "other_info": other_info}

    date = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

    with open(os.path.join(saving_path, f"metadata_{mouse_id}_{date}.json"),
              "w", encoding="utf-8") as outfile:
        json.dump(metadata, outfile)

    print("\n\n- - Starting acquisition - - ")

    with open(os.path.join(saving_path, f"data_{mouse_id}_{date}.txt"), "w",
              encoding="utf-8") as file:
        file.write("time\tarduino_output\n")

        fig, ax = plt.subplots(1)
        line_led0, = ax.plot([], [], label="Isobestic", c="m")
        line_led1, = ax.plot([], [], label="Fluorescence", c="b")

        led0_signal = np.zeros(100) # 405 nm
        led1_signal = np.zeros(100) # 470 nm
        plt.ylim(10000, 1e12)
        plt.xlim(0, 100)
        plt.xticks([0, 20, 40, 60, 80, 100], range(6))
        plt.xlabel("Time [s]")
        plt.ylabel("Intensity [a.u.]")
        plt.legend()
        plt.pause(0.01)

        arduino.reset_input_buffer()
        arduino.write(b"3")

        print(f"Unlimited acquisition started. Close the graph and type KeyboardInterrupt (ctrl-C)")

        try:
            while True:
                data = (arduino.readline()).decode('utf-8')

                try:
                    if int(data[0]) == 1:
                        led0_signal = np.roll(led0_signal, -1)
                        led0_signal[1] = int(data[-12:])

                    if int(data[1]) == 1:
                        led1_signal = np.roll(led1_signal, -1)
                        led1_signal[1] = int(data[-12:])

                except BaseException:
                    pass

                line_led0.set_data(range(100), led0_signal)
                line_led1.set_data(range(100), led1_signal)

                file.write(f"{time.perf_counter_ns()}\t{data}")
                plt.pause(0.0001)

        except KeyboardInterrupt:
            arduino.write(b"0")
            print("- - - Ending Fused Fiber Photometry Software - - -")



if __name__ == "__main__":
    print("- - - Starting Wide Field Software - - -\n\n")
    print("List of the active USB ports:")

    for port, desc, hwid in sorted(comports()):
        print(f"\t- {port}: {desc}")

    daq_port = input("\n\nSelect the USB port of the WF DAQ: ")

    arduino = serial.Serial(
                        port=daq_port,
                        baudrate=19200,
                        timeout=1)

    
    acquisition_type = input("Do you want a test (1), a timed (2) or an unlimited (3) acquisition? ")



    if acquisition_type == "1":
        test_acquisition()

    if acquisition_type == "2":
        timed_acquisition()

    if acquisition_type == "3":
        unlimited_acquisition()