from matplotlib import pyplot as plt
from matplotlib import colormaps as clm
import numpy as np
# from prepData import 
from scipy.stats import sem
import seaborn as sns
import seaborn_image as snsi
import cmcrameri.cm as cmc
import os
import cv2

data_path = r"D:\ggermain\2025-03-21_opto_M914\1_whiskerpad"
computed_folder = "gcamp_computed"


files_list = os.listdir(os.path.join(data_path, computed_folder))

# print(np.arange(30, 1000, 32))

for idx, file in enumerate(files_list):
    path = os.path.join(data_path, computed_folder, file)
    data = np.load(path)

    if idx == 0:
        timeseries = np.zeros((len(files_list), data.shape[0]))   # moyenne sur une région, temporel seulement
        x_mot, y_mot, w_mot, h_mot = cv2.selectROI("Select motion zone", data[0,:,:])
        cv2.destroyAllWindows()
    
    avg_zone = data[:,y_mot:y_mot+h_mot, x_mot:x_mot+w_mot]
    avg = np.mean(avg_zone, axis=(1, 2))

    avg = (avg-np.min(avg))/(np.max(avg)-np.min(avg))

    timeseries[idx,:] = avg

plt.imshow(timeseries, origin='lower', extent=[-3, 10, 0, len(files_list)])
plt.show()

avg_data = np.mean(timeseries, axis=0)
std_data = sem(timeseries, axis=0)

plt.plot(np.linspace(-3, 10, 130), avg_data)
plt.fill_between(np.linspace(-3, 10, 130), avg_data-std_data, avg_data+std_data, alpha=0.2)
plt.show()


timestamps = np.linspace(-3, 10, timeseries.shape[1])

# titles = ("Oxygenated hemoglobin [HbO]", "Deoxygenated hemoglobin [HbR]", "Total hemoglobin [HbT]")
# cols = ('red', 'royalblue', 'green')
# cmaps = (clm['Reds_r'], clm['Blues_r'], clm['Greens_r'])

# def raster_avg(data, timestamps, cmaps, titles):
    # fig, axs = plt.subplots(3, 2, figsize=(14,7), width_ratios=[5, 4])
    # sns.set_context('notebook')

#     for idx, (sig, cmap, title) in enumerate(zip(aligned_data, cmaps, titles)):
#         ax = plt.subplot(3, 2, 2*idx+1)
#         ax.set_title(title)
#         ax.set_xlabel('time relative to airpuff [s]')
#         ax.set_ylabel("trial [-]")
#         pos = ax.imshow(aligned_data_norm[idx,:,:], origin='lower', extent=[time[AP_idx[0]-inf]-time[AP_idx[0]], time[AP_idx[0]+sup]-time[AP_idx[0]], 0, ncycles], aspect='auto', cmap=cmap)
#         fig.colorbar(pos, ax=ax)

# for idx, (sig, col, title) in enumerate(zip(aligned_data_raw, cols, titles)):
#     avg_data = np.mean(sig, axis=0)
#     std_data = sem(sig, axis=0)

#     ax = plt.subplot(3, 2, 2*idx+2)
#     ax.set_title(title)
#     # ax.vlines(0, avg_data.min(), avg_data.max(), color='grey', linestyles='-')
#     ax.plot(time[AP_idx[0]-inf:AP_idx[0]+sup]-time[AP_idx[0]], avg_data, color=col)
#     ax.fill_between(time[AP_idx[0]-inf:AP_idx[0]+sup]-time[AP_idx[0]], avg_data-std_data, avg_data+std_data, color=col, alpha=0.2)
#     ax.set_xlim(time[AP_idx[0]-inf]-time[AP_idx[0]], time[AP_idx[0]+sup]-time[AP_idx[0]])
#     ax.set_xlabel('time relative to airpuff [s]')
#     if idx == 1:
#         ax.set_ylabel("Concentration variation [uM]")
#     # sns.despine()
#     plt.tight_layout()