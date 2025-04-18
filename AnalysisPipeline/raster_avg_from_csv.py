#%%
from matplotlib import pyplot as plt
from matplotlib import colormaps as clm
import numpy as np
from scipy.stats import sem
from scipy.ndimage import median_filter
import seaborn as sns
import seaborn_image as snsi
import cmcrameri.cm as cmc
import os
import cv2
from tqdm import tqdm


def normalise(data):
    dmax = np.max(data)
    dmin = np.min(data)

    return (data-dmin)/(dmax-dmin)

#%% --- open data ---

attente = 30
stim_dur = 5                        # changer
Ns_bef = 3
Ns_aft = 13
freq = 50

    # all
# titles = ("GCaMP", "HbO", "HbR", "HbT", "LSCI", "Face motion", "Pupil")
# cols = ("tab:green", "tab:red", "tab:blue", "darkred", "tab:purple", "tab:orange", "royalblue")
# cmaps = (clm['Greens_r'], clm['Reds_r'], clm['Blues_r'], clm['Reds_r'], clm["Purples_r"], clm["Oranges_r"], clm['Blues_r'])
    # vasc
# titles = ("HbO", "HbR", "HbT", "LSCI", "Face motion", "Pupil")
# cols = ("tab:red", "tab:blue", "darkred", "tab:purple", "tab:orange", "royalblue")
# cmaps = (clm['Reds_r'], clm['Blues_r'], clm['Reds_r'], clm["Purples_r"], clm["Oranges_r"], clm['Blues_r'])
    # GCAMP + HBT + face
titles = ("GCaMP", "HbT",  "Face motion", "Pupil")
cols = ("tab:green",  "darkred",  "tab:orange", "royalblue")
cmaps = (clm['Greens_r'], clm['Reds_r'], clm["Oranges_r"], clm['Blues_r'])

event_times = np.arange(attente, 2000, attente+stim_dur)              # if opto
first_stim = 30 

# event_times = np.load(r"C:\Users\gabri\Documents\Université\Maitrise\Projet\Widefield-Imaging-Acquisition\AnalysisPipeline\Air_puff_timestamps.npy")        # if airpuffs
# first_stim = 12.01  # Air puffs


# path = r"D:\ggermain\2024-09-17_air_puffs"
# path = r"D:\ggermain\2025-03-21_opto2s\1_whiskerpad"
# path = r"D:\ggermain\2025-03-21_opto2s\2_pupil"
# path = r"D:\ggermain\2025-04-02_opto5s\1_rideau_ferme"
path = r"D:\ggermain\2025-04-02_opto5s\2_rideau_ouvert"


gcamp = np.loadtxt(os.path.join(path, "470_3.csv"), skiprows=1, delimiter=',', usecols=1) - np.loadtxt(os.path.join(path, "405_3.csv"), skiprows=1, delimiter=',', usecols=1)
# gcamp = np.loadtxt(os.path.join(path, "470_3.csv"), skiprows=1, delimiter=',', usecols=1)           # 17 sept
HbO, HbR, HbT = np.load(os.path.join(path, "computedHb_ts_3.npy"))
LSCI = np.loadtxt(os.path.join(path, "LSCI_3.csv"), skiprows=1, usecols=1, delimiter=',') - np.loadtxt(os.path.join(path, "LSCI_static.csv"), skiprows=1, usecols=1, delimiter=',')

face_motion = median_filter(np.load(os.path.join(path, "face_motion.npy"), allow_pickle=True), size=5)
pupil = median_filter(np.load(os.path.join(path, "pupil.npy"), allow_pickle=True).item()['pupil'][0]['area_smooth'], size=5)

# all
# unaligned_data = [gcamp, HbO, HbR, HbT, LSCI, face_motion, pupil]                      # changer
# vasc
# unaligned_data = [HbO, HbR, HbT, LSCI]
# gcamp + hbt + face
unaligned_data = [gcamp, HbT, face_motion, pupil]


timestamps_wf = np.sort(np.load(path + "\\470ts.npy"))
timestamps_bh = np.sort(np.array([timestamps_wf, timestamps_wf, timestamps_wf, timestamps_wf, timestamps_wf]).flatten())
timestamps_bh[1::5] += 1/freq
timestamps_bh[2::5] += 2/freq
timestamps_bh[3::5] += 3/freq
timestamps_bh[4::5] += 4/freq
timestamps_bh = timestamps_bh*1.00025
n_frames_wf = len(timestamps_wf)
n_frames_bh = len(timestamps_bh)
max_time = timestamps_bh[-1]
last_event = np.argmin(np.abs((event_times - max_time)))
event_times = event_times[0:last_event] # always skip last event, most of the time incomplete



timestamps = [timestamps_wf, timestamps_wf, timestamps_bh, timestamps_bh]            # changer


#%% --- sort and align data ---

sigs = []
for idx_d, data in enumerate(unaligned_data):
    
    for idx_e, event in enumerate(event_times):
        
        if idx_d == 2 or idx_d == 3:      # behavior                  #changer
            Nf_bef = Ns_bef*freq
            Nf_aft = Ns_aft*freq
            if idx_e == 0:
                sig = np.zeros([len(event_times), (Nf_bef+Nf_aft)])                
            stim_idx = np.argmin(np.absolute(timestamps_bh-event))

        else:                # wide field
            Nf_bef = Ns_bef*(freq//5)
            Nf_aft = Ns_aft*(freq//5)
            if idx_e == 0:
                sig = np.zeros([len(event_times), (Nf_bef+Nf_aft)])
            stim_idx = np.argmin(np.absolute(timestamps_wf-event))
            # print(timestamps_wf[stim_idx-Nf_bef:stim_idx+Nf_aft])
        data_stim = data[stim_idx-Nf_bef:stim_idx+Nf_aft]
        data_stim = normalise(data_stim)
        sig[idx_e,:] = data_stim

    sigs.append(sig)



#%%
timestamps_wf = timestamps_wf[(np.argmin(np.absolute(timestamps_wf-first_stim)))-(Ns_bef*freq//5):(np.argmin(np.absolute(timestamps_wf-first_stim)))+sigs[0].shape[1]-(Ns_bef*freq//5)] # -30 pour les 3 sec avant
timestamps_wf -= first_stim
timestamps_bh = timestamps_bh[(np.argmin(np.absolute(timestamps_bh-first_stim)))-(Ns_bef*freq):(np.argmin(np.absolute(timestamps_bh-first_stim)))+sigs[-1].shape[1]-(Ns_bef*freq)]
timestamps_bh -= first_stim


timestamps = [timestamps_wf, timestamps_wf, timestamps_bh, timestamps_bh]              # changer


#%% --- figure ---

fig, axs = plt.subplots(len(sigs), 2, figsize=(14,7), width_ratios=[5, 4])
sns.set_context('notebook')

# raster plots
for idx, (sig, cmap, title, timestamp) in enumerate(zip(sigs, cmaps, titles, timestamps)):
    ax = plt.subplot(len(sigs), 2, 2*idx+1)
    ax.set_title(title)
    if idx == (len(sigs)-1):
        ax.set_xlabel('time relative to airpuff [s]')
    if idx == (len(sigs)//2):
        ax.set_ylabel("trial [-]")
    raster = ax.imshow(sig, origin='lower', extent=[timestamp[0], timestamp[-1], 0, sig.shape[0]], aspect='auto', cmap=cmap)
    fig.colorbar(raster, ax=ax)

# average plots
for idx, (sig, col, title, timestamp) in enumerate(zip(sigs, cols, titles, timestamps)):
    avg_data = np.mean(sig, axis=0)
    std_data = sem(sig, axis=0)
    # print("--- ", title, " ---")
    # print("Min à {:.3f} s".format(timestamp[np.argmin(avg_data)]))
    # print("Max à {:.3f} s".format(timestamp[np.argmax(avg_data)]))

    ax = plt.subplot(len(sigs), 2, 2*idx+2)
    ax.set_title(title)
    ax.fill_between((0, stim_dur), 0, 1, color="grey", alpha=0.2, label="stim")   # opto seulement
    ax.plot(timestamp, avg_data, color=col)
    ax.fill_between(timestamp, avg_data-std_data, avg_data+std_data, color=col, alpha=0.2)
    ax.set_xlim(timestamp[0], timestamp[-1])
    ax.set_ylim(avg_data.min() - std_data.max(), avg_data.max() + std_data.max())
    ax.set_yticklabels([])
    ax.set_yticks([])
    if idx == (len(sigs)-1):
        ax.set_xlabel('time relative to airpuff [s]')
    if idx == (len(sigs)//2):
        ax.set_ylabel("average signal [normalized]")

sns.despine()
plt.tight_layout()
# plt.savefig("name.png", dpi=600)
plt.savefig("04-02_ouvert_varie.svg")
# plt.show()

# %%
