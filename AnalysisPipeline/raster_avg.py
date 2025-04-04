from matplotlib import pyplot as plt
from matplotlib import colormaps as clm
import numpy as np
from scipy.stats import sem
import seaborn as sns
import seaborn_image as snsi
import cmcrameri.cm as cmc
import os
import cv2
from tqdm import tqdm



# chose data, à choisir à la main

Ns_aft = 15
data_path = r"D:\\ggermain\\2025-04-02_opto5s\\1" 
# data_path = r"D:\\ggermain\\2025-03-21_opto2s\\1_whiskerpad"            # expérience: date, essai
computed_folders = ( "GCaMP", "LSCI") #, "Hb" "pupil", "motion_energy")     # data à comparer, choisir data appropprié

# for figures, make suredimensions are compatible with signals
titles = computed_folders
cols = ('red', 'royalblue', 'green')
cmaps = (clm['Reds_r'], clm['Blues_r'], clm['Greens_r'])


#-----------------------------------------------


# open and sort data
files_array = []
for folder in computed_folders:
    files_array.append(os.listdir(os.path.join(data_path, "computed_npy", folder)))

sigs = []
for files_list, folder in zip(files_array, computed_folders):

    if folder == "Hb":
        pass #gérer le fait que Hb a une dimension de plus (HbO, HbR, HbT)

    for idx, file in tqdm(enumerate(files_list)):
        path = os.path.join(data_path, "computed_npy", folder, file)
        data = np.load(path)

        if idx == 0:
            timeseries = np.zeros((len(files_list), data.shape[0]))   # moyenne sur une région, temporel seulement, 2D
            x_mot, y_mot, w_mot, h_mot = cv2.selectROI("Select zone", data[0,:,:])
            cv2.destroyAllWindows()
        
        avg_zone = data[:,y_mot:y_mot+h_mot, x_mot:x_mot+w_mot]
        avg = np.mean(avg_zone, axis=(1, 2))
        avg = avg - np.mean(avg[:30]) # on veut soustraire le baseline avant la stim #(avg-np.min(avg))/(np.max(avg)-np.min(avg))
        timeseries[idx,:] = avg

    sigs.append(timeseries)

timestamps = [np.linspace(-3, Ns_aft, len(sigs[0][0])), np.linspace(-3, Ns_aft, len(sigs[1][0]))] #, np.linspace(-3, Ns_aft, len(sigs[2]))] # rendre robuste pour différentes longueurs d'analyses

# figure
fig, axs = plt.subplots(len(sigs), 2, figsize=(14,7), width_ratios=[5, 4])
sns.set_context('notebook')

# raster plots
for idx, (sig, cmap, title, timestamp) in enumerate(zip(sigs, cmaps, titles, timestamps)):
    ax = plt.subplot(len(sigs), 2, 2*idx+1)
    ax.set_title(title)
    ax.set_xlabel('time relative to airpuff [s]')
    ax.set_ylabel("trial [-]")
    raster = ax.imshow(sig, origin='lower', extent=[timestamp[0], timestamp[-1], 0, sig.shape[0]], aspect='auto', cmap=cmap)
    fig.colorbar(raster, ax=ax)

# average plots
for idx, (sig, col, title, timestamp) in enumerate(zip(sigs, cols, titles, timestamps)):
    avg_data = np.mean(sig, axis=0)
    std_data = sem(sig, axis=0)

    ax = plt.subplot(len(sigs), 2, 2*idx+2)
    ax.set_title(title)
    ax.vlines(0, avg_data.min(), avg_data.max(), color='grey', linestyles='-')
    ax.plot(timestamp, avg_data, color=col)
    ax.fill_between(timestamp, avg_data-std_data, avg_data+std_data, color=col, alpha=0.2)
    ax.set_xlim(timestamp[0], timestamp[-1])
    ax.set_xlabel('time relative to airpuff [s]')
    # if idx == 1:
        # ax.set_ylabel("Concentration variation [uM]")

sns.despine()
plt.tight_layout()
# plt.savefig("name.svg")
# plt.savefig("name.png", dpi=600)
plt.show()