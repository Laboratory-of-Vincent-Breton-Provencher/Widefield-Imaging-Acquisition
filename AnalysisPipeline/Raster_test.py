import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from tkinter import filedialog, Tk
import os

# --- Sélection du dossier via boîte de dialogue ---
root = Tk()
root.withdraw()
data_path = filedialog.askdirectory(title="Sélectionne le dossier de données")
if not data_path:
    raise Exception("Aucun dossier sélectionné.")

# --- Chargement du fichier unique ---
gcamp_file = os.path.join(data_path, "computedLSCI.npy")
data = np.load(gcamp_file)  # (n_frames, y, x)

# --- Moyenne sur tout le frame ---
trace = data.mean(axis=(1,2))  # (n_frames,)

# --- Normalisation min-max ---
trace_norm = (trace - trace.min()) / (trace.max() - trace.min())

# --- Chargement des timestamps ---
ts_file = os.path.join(data_path, "785ts.npy")
ts = np.load(ts_file)
if len(ts) != trace.shape[0]:
    raise Exception("470ts.npy ne correspond pas au nombre de frames.")

# --- Timestamps connus des stimulations ---
stim_16_7 = [20, 86, 152, 218, 284]
stim_8_3  = [42, 108, 174, 240, 306]
stim_4_16 = [64, 130, 196, 262, 328]
all_stims = stim_16_7 + stim_8_3 + stim_4_16

stim_dur = 2
window_before = 5
window_after = 10

def extract_trials(trace, ts, stim_times, window_before, stim_dur, window_after):
    trials = []
    t_windows = []
    min_len = None
    for stim_time in stim_times:
        idx_start = np.searchsorted(ts, stim_time - window_before)
        idx_end = np.searchsorted(ts, stim_time + stim_dur + window_after)
        t_window = ts[idx_start:idx_end] - stim_time
        trace_window = trace[idx_start:idx_end]
        if min_len is None or len(trace_window) < min_len:
            min_len = len(trace_window)
        trials.append(trace_window)
        t_windows.append(t_window)
    # Tronque toutes les fenêtres à la même longueur (la plus courte)
    trials = [tr[:min_len] for tr in trials]
    t_windows = [tw[:min_len] for tw in t_windows]
    return np.array(trials), np.array(t_windows[0])

trials_16_7, t_win_16_7 = extract_trials(trace_norm, ts, stim_16_7, window_before, stim_dur, window_after)
trials_8_3,  t_win_8_3  = extract_trials(trace_norm, ts, stim_8_3,  window_before, stim_dur, window_after)
trials_4_16, t_win_4_16 = extract_trials(trace_norm, ts, stim_4_16, window_before, stim_dur, window_after)

# --- Figure combinée ---
fig = plt.figure(figsize=(18, 10))

# 1. Signal global avec axes verticaux colorés
ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax0.plot(ts, trace_norm, color='green', label='ΔF/F')

colors = ['forestgreen', 'royalblue', 'orange']
for stim in stim_16_7:
    ax0.axvline(stim, color=colors[0], linestyle='--', alpha=0.7, label="16.7 Hz" if stim == stim_16_7[0] else "")
for stim in stim_8_3:
    ax0.axvline(stim, color=colors[1], linestyle='--', alpha=0.7, label="8.3 Hz" if stim == stim_8_3[0] else "")
for stim in stim_4_16:
    ax0.axvline(stim, color=colors[2], linestyle='--', alpha=0.7, label="4.16 Hz" if stim == stim_4_16[0] else "")

ax0.set_xlabel("Temps (s)")
ax0.set_ylabel("ΔF/F [-]")

# Légende sans doublons
handles, labels = ax0.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax0.legend(by_label.values(), by_label.keys())

# 2. Raster et moyenne ± SEM pour chaque fréquence
freqs = [16.7, 8.3, 4.16]
trial_sets = [trials_16_7, trials_8_3, trials_4_16]
t_windows = [t_win_16_7, t_win_8_3, t_win_4_16]

for i, (freq, trials, t_win, col) in enumerate(zip(freqs, trial_sets, t_windows, colors)):
    # Raster
    ax_raster = plt.subplot2grid((3, 3), (1, i))
    im = ax_raster.imshow(trials, aspect='auto', extent=[t_win[0], t_win[-1], 0, 5], origin='lower', cmap='Greens')
    ax_raster.axvspan(0, stim_dur, color=col, alpha=0.2)
    ax_raster.set_title(f"{freq} Hz")
    ax_raster.set_xlabel("Temps relatif à la stim (s)")
    ax_raster.set_ylabel("Essai")
    plt.colorbar(im, ax=ax_raster, label="ΔF/F")

    # Moyenne ± SEM
    ax_avg = plt.subplot2grid((3, 3), (2, i))
    avg = trials.mean(axis=0)
    std = sem(trials, axis=0)
    ax_avg.axvspan(0, stim_dur, color=col, alpha=0.2)
    ax_avg.plot(t_win, avg, color=col, label=f"{freq} Hz")
    ax_avg.fill_between(t_win, avg-std, avg+std, color=col, alpha=0.3)
    ax_avg.set_xlabel("Temps relatif à la stim (s)")
    ax_avg.set_ylabel("ΔF/F")
    ax_avg.legend()

plt.tight_layout()
plt.show()