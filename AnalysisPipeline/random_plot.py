#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter
import seaborn as sns
from scipy.stats import sem

def normalise(data):
    dmax = np.max(data)
    dmin = np.min(data)

    return (data-dmin)/(dmax-dmin)


#%% --- Lumière IR behavior ---

# path = r"D:\ggermain\2025-03-21_opto2s\ir_grey_value.csv"
# _, data = np.loadtxt(path, skiprows=1, delimiter=',').transpose()
# path = r"D:\ggermain\2025-03-21_opto2s\3_ir_test_ts.npy"
# ts = np.load(path)
# # print(1/np.mean(np.diff(ts)))
# plt.plot(ts, data, color='k')
# plt.xlabel("Temps (s)")
# plt.ylabel("Intensité du signal (grey value)")
# plt.text(3, 250, "Lumière allumée")
# plt.text(13.5, 250, "Lumière éteinte")
# # plt.fill_between((0, 12.168), 100, 400, color='grey', alpha=0.3)
# plt.fill_between((12.168, 21.311), 100, 400, color='grey', alpha=0.3)
# # plt.fill_between((21.311, 30), 100, 400, color='g', alpha=0.3)
# plt.xlim(ts[0], ts[-1])
# plt.ylim(100, 400)
# # plt.savefig("ir_behavior_effet.png", dpi=600)
# plt.show()

#%% --- Subsampling de 60 Hz ---

# x = np.linspace(0, 20, 5000)
# plt.plot(x, np.sin(60*x))
# plt.plot(x, np.sin(11.74*x))

# x9 = np.linspace(0, 20, 200)
# plt.plot(x9, np.sin(60*x9), 'g.')
# plt.show()

#%% --- pupille rideau ouvert vs fermé ---

# n = 3
# data1 = median_filter(np.load(r"D:\\ggermain\\2025-04-02_opto5s\\1_rideau_ferme\\pupil.npy", allow_pickle=True).item()['pupil'][0]['area_smooth'], size=n)
# data2 = median_filter(np.load(r"D:\\ggermain\\2025-04-02_opto5s\\2_rideau_ouvert\\pupil.npy", allow_pickle=True).item()['pupil'][0]['area_smooth'], size=n)
# data1 = data1[:len(data2)]

# freq = 50
# timestamps_wf = np.sort(np.load(r"D:\\ggermain\2025-04-02_opto5s\2_rideau_ouvert\470ts.npy"))
# timestamps_bh = np.sort(np.array([timestamps_wf, timestamps_wf, timestamps_wf, timestamps_wf, timestamps_wf]).flatten())
# timestamps_bh[1::5] += 1/freq
# timestamps_bh[2::5] += 2/freq
# timestamps_bh[3::5] += 3/freq
# timestamps_bh[4::5] += 4/freq
# timestamps_bh = timestamps_bh*1.00025
# timestamps_bh = timestamps_bh[:-1]

# attente = 30
# stim_dur = 5
# event_times = np.arange(attente, timestamps_bh[-1], attente+stim_dur)

# plt.plot(timestamps_bh, data1, label="Rideau fermé")
# plt.plot(timestamps_bh, data2, label="Rideau ouvert")
# plt.vlines(event_times, 0, 400, color='grey')
# plt.legend()
# plt.show()

# data3 = median_filter(np.load(r"D:\ggermain\2025-03-21_opto2s\2_pupil\pupil.npy", allow_pickle=True).item()['pupil'][0]['area_smooth'], size=n)

# freq = 50
# timestamps_wf = np.sort(np.load(r"D:\ggermain\2025-03-21_opto2s\2_pupil\470ts.npy"))
# timestamps_bh = np.sort(np.array([timestamps_wf, timestamps_wf, timestamps_wf, timestamps_wf, timestamps_wf]).flatten())
# timestamps_bh[1::5] += 1/freq
# timestamps_bh[2::5] += 2/freq
# timestamps_bh[3::5] += 3/freq
# timestamps_bh[4::5] += 4/freq
# timestamps_bh = timestamps_bh*1.00025
# data3 = data3[:-1]


# stim_dur = 2
# event_times = np.arange(attente, 330, attente+stim_dur)

# plt.plot(timestamps_bh, data3)
# plt.vlines(event_times, 0, timestamps_bh[-1], color="grey")
# plt.show()

#%% --- Gcamp avec/sans isobestique

# _405 = np.loadtxt(r"D:\ggermain\2025-03-21_opto2s\2_pupil\405.csv", skiprows=1, usecols=1, delimiter=',').transpose()
# _470 = np.loadtxt(r"D:\ggermain\2025-03-21_opto2s\2_pupil\470.csv", skiprows=1, usecols=1, delimiter=',').transpose()
# ts = np.load(r"D:\ggermain\2025-03-21_opto2s\2_pupil\405ts.npy")

# attente = 30
# stim_dur = 2
# event_times = np.arange(attente, ts[-1], attente+stim_dur)


# mul = 1.3
# fig, axs = plt.subplots(2, 1, figsize=(7, 5))
# ax1 = plt.subplot(2, 1, 1)
# ax1.plot(ts, _405*mul, color="tab:purple", label="405 nm raw")
# ax1.plot(ts, _470, color='tab:blue', label='470 nm raw')
# ax1.plot(ts, _470-(_405*mul)+450, color='tab:green', label="470 nm - 405 nm")
# ax1.legend()
# ax1.set_ylabel("signal intensity [a.u.] ")
# ax1.set_yticklabels([])
# ax1.set_yticks([])
# ax1.set_title('a)', loc='left')
# plt.vlines([80, 150], 400, 1400, color="grey", linestyles='--')
# plt.hlines([400, 1400], 80, 150, color='grey', linestyles='--')

# ax2 = plt.subplot(2, 1, 2)
# ax2.plot(ts, _405*mul, color="tab:purple")
# ax2.plot(ts, _470, color='tab:blue')
# ax2.plot(ts, _470-(_405*mul)+450, color='tab:green')
# ax2.set_xlim(80, 150)
# ax2.set_ylabel("signal intensity [a.u.] ")
# ax2.set_yticklabels([])
# ax2.set_yticks([])
# ax2.set_xlabel("time [s]")
# ax2.set_title('b)', loc='left')

# sns.despine()
# plt.tight_layout()
# plt.savefig("isosbestic.svg")
# plt.show()

#%% --- 4 rois air puffs ---

titles = ("HbO", "HbR", "HbT", "LSCI")
cols = ("tab:blue", "tab:red", "tab:green")
rois = ("veine", "artériole", "parenchyme")
 
Ns_bef = 2
Ns_aft = 5
freq = 50

path = r"D:\ggermain\2024-09-17_air_puffs"

event_times = np.load(r"C:\Users\gabri\Documents\Université\Maitrise\Projet\Widefield-Imaging-Acquisition\AnalysisPipeline\Air_puff_timestamps.npy")        # if airpuffs
first_stim = 12.01  # Air puffs

fig, axs = plt.subplots(4, 1, figsize=(6, 6))
sns.set_context('notebook')

Nf_bef = Ns_bef*(freq//5)
Nf_aft = Ns_aft*(freq//5)

for N in range(3):
    print(rois[N])
    N+=1

    HbO, HbR, HbT = np.load(os.path.join(path, "computedHb_ts_{}.npy".format(N)))
    LSCI = np.loadtxt(os.path.join(path, "LSCI_{}.csv".format(N)), usecols=1, skiprows=1, delimiter=',').transpose() - np.loadtxt(os.path.join(path, "LSCI_static.csv".format(N)), usecols=1, skiprows=1, delimiter=',').transpose()
    ts = np.load(os.path.join(path, "470ts.npy"))
    unaligned_data = [HbO, HbR, HbT, LSCI]

    sigs = []
    for idx_d, data in enumerate(unaligned_data):
        for idx_e, event in enumerate(event_times):
            if idx_e == 0:
                sig = np.zeros([len(event_times), (Nf_bef+Nf_aft)])
            stim_idx = np.argmin(np.absolute(ts-event))
            data_stim = data[stim_idx-Nf_bef:stim_idx+Nf_aft]
            data_stim = normalise(data_stim)
            sig[idx_e,:] = data_stim
        sigs.append(sig)

    ts = ts[(np.argmin(np.absolute(ts-first_stim)))-(Ns_bef*freq//5):(np.argmin(np.absolute(ts-first_stim)))+sigs[0].shape[1]-(Ns_bef*freq//5)] # -30 pour les 3 sec avant
    ts -= first_stim

    for idx, (sig, title) in enumerate(zip(sigs, titles)):
        avg_data = np.mean(sig, axis=0)
        std_data = sem(sig, axis=0)
        if idx == 0 or idx == 2 or idx == 3:
            print(titles[idx], "lag: ", ts[np.argmax(avg_data)], ' s')

        if idx == 1:
            print(titles[idx], "lag: ", ts[np.argmin(avg_data)], ' s')
        
        ax = plt.subplot(4, 1, idx+1)
        ax.set_title(titles[idx])
        ax.plot(ts, avg_data, label=rois[N-1], color=cols[N-1])
        ax.fill_between(ts, avg_data-std_data, avg_data+std_data, color=cols[N-1], alpha=0.2)
        if idx == 0:
            ax.legend(loc='right')
        ax.set_yticklabels([])
        ax.set_yticks([])
        if idx == 3:
            ax.set_xlabel("time [s]", fontsize=12)
        if idx == 1:
            ax.set_ylabel("signal intensity [normalised]", fontsize=12)
        ax.set_xlim(ts[0], ts[-1])

sns.despine()
plt.tight_layout()
plt.savefig("09-17_airpuffs_rois.svg")
# plt.show()
# %%
