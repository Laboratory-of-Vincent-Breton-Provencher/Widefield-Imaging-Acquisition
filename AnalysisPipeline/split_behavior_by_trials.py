import numpy as np
import os
from tkinter import filedialog
from tkinter import *
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *


# Parameters

pupil = 0
motion_energy = 1

attente = 30
stim = 5 
Ns_bef = 3
Ns_aft = 15
freq = 50
event_times = np.load(r"AnalysisPipeline\Air_puff_timestamps.npy")        # if airpuffs
# event_times = np.arange(attente, 2000, attente+stim)            # if otogenetics


# ------------------------------------------


root = Tk()
root.withdraw()
path = filedialog.askdirectory()
timestamps = np.sort(np.concatenate((np.load(path + "\\405ts.npy"), 
                        np.load(path + "\\470ts.npy"),
                        np.load(path + "\\530ts.npy"),
                        np.load(path + "\\625ts.npy"),
                        np.load(path + "\\785ts.npy"))))

# plt.plot(timestamps)
timestamps[1::5] += 1/freq
timestamps[2::5] += 2/freq
timestamps[3::5] += 3/freq
timestamps[4::5] += 4/freq
# plt.plot(timestamps)
# plt.show()

np.save(path + "\\behaviorts.npy", timestamps)

sigs = []
names = []
if pupil:
    pupil = np.load(path + "\\pupil.npy", allow_pickle=True).item()['pupil'][0]['area_smooth']
    sigs.append(pupil)
    names.append("pupil")
if motion_energy:
    motion_energy = np.load(path + "\\face_motion.npy")
    sigs.append(motion_energy)
    names.append("motion_energy")



n_frames = len(timestamps)
max_time = timestamps[n_frames-1]
last_event = np.argmin(np.abs((event_times - max_time)))
event_times = event_times[0:last_event+1]


Nf_bef = Ns_bef*freq
Nf_aft = Ns_aft*freq
for (sig, name) in zip(sigs, names):
    try:
        os.mkdir(path + "\\computed_npy")
    except FileExistsError:
        pass
    try:
        os.mkdir(path + "\\computed_npy\\{}".format(name))
    except FileExistsError:
        pass

    for idx, event in enumerate(event_times):
        stim_idx = np.argmin(np.absolute(timestamps-event))
        data_stim =  sig[stim_idx-Nf_bef:stim_idx+Nf_aft]

        np.save(path + "\\computed_npy\\{}\\computed{}_trial{}.npy".format(name, name, idx+1), data_stim)

    # plt.plot(data_stim)
    # plt.show()


