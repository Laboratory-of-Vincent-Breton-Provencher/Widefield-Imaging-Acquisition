import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from tkinter import filedialog, Tk
import os

# --- Choix des figures à générer ---
DO_GCAMP = False
DO_HB = False
DO_LSCI = False
DO_FACE = False
DO_PUPIL = True

# --- Sélection du dossier via boîte de dialogue ---
root = Tk()
root.withdraw()
data_path = filedialog.askdirectory(title="Sélectionne le dossier de données")
if not data_path:
    raise Exception("Aucun dossier sélectionné.")

# --- Création du dossier Figures si inexistant ---
figures_path = os.path.join(data_path, "Figures")
os.makedirs(figures_path, exist_ok=True)

# Paramètres communs
baseline_init = 2 * 60  # 2 min en secondes
stim_dur = 2            # 2 s
baseline_between = 30   # 20 s
cycles = 10              # nombre de stimulations par fréquence

freqs = [4.16, 8.3, 16.6]
colors = ['forestgreen', 'royalblue', 'orange']

stim_16_7 = []
stim_8_3 = []
stim_4_16 = []

window_before = 5
window_after = 15

t = baseline_init
for i in range(cycles):
    stim_4_16.append(t)
    t += stim_dur + baseline_between
    stim_8_3.append(t)
    t += stim_dur + baseline_between
    stim_16_7.append(t)
    t += stim_dur + baseline_between

stim_sets = [stim_4_16, stim_8_3, stim_16_7]


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
    trials = [tr[:min_len] for tr in trials]
    t_windows = [tw[:min_len] for tw in t_windows]
    return np.array(trials), np.array(t_windows[0])

# --- GCaMP ---
if DO_GCAMP:
    gcamp_file = os.path.join(data_path, "computedGCaMP.npy")
    gcamp_ts_file = os.path.join(data_path, "470ts.npy")
    if os.path.exists(gcamp_file) and os.path.exists(gcamp_ts_file):
        data = np.load(gcamp_file)  # (n_frames, y, x)
        ts = np.load(gcamp_ts_file)
        min_len = min(data.shape[0], len(ts))
        if data.shape[0] != len(ts):
            print(f"Attention: computedGCaMP.npy ({data.shape[0]}) et 470ts.npy ({len(ts)}) n'ont pas le même nombre de frames. Tronquage à {min_len}.")
            data = data[:min_len]
            ts = ts[:min_len]
        trace = data.mean(axis=(1,2))
        trace_norm = (trace - trace.min()) / (trace.max() - trace.min())
        trial_sets = []
        t_windows = []
        for stim_times in stim_sets:
            trials, t_win = extract_trials(trace_norm, ts, stim_times, window_before, stim_dur, window_after)
            trial_sets.append(trials)
            t_windows.append(t_win)
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("Raster - GCaMP")
        ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
        ax0.plot(ts, trace_norm, color='green', label='GCaMP')
        for stim, col, label in zip(stim_sets, colors, [f"{f} Hz" for f in freqs]):
            for s in stim:
                ax0.axvline(s, color=col, linestyle='--', alpha=0.7, label=label if s == stim[0] else "")
        ax0.set_xlabel("Temps (s)")
        ax0.set_ylabel("Signal normalisé")
        handles, labels_ = ax0.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        ax0.legend(by_label.values(), by_label.keys())
        for i, (freq, trials, t_win, col) in enumerate(zip(freqs, trial_sets, t_windows, colors)):
            ax_raster = plt.subplot2grid((3, 3), (1, i))
            im = ax_raster.imshow(trials, aspect='auto', extent=[t_win[0], t_win[-1], 0, trials.shape[0]],
                                  origin='lower', cmap='Greens', interpolation='none')
            ax_raster.axvspan(0, stim_dur, color=col, alpha=0.2)
            ax_raster.set_title(f"{freq} Hz")
            ax_raster.set_xlabel("Temps relatif à la stim (s)")
            ax_raster.set_ylabel("Essai")
            plt.colorbar(im, ax=ax_raster, label="Signal norm.")
            ax_avg = plt.subplot2grid((3, 3), (2, i))
            avg = trials.mean(axis=0)
            std = sem(trials, axis=0)
            ax_avg.axvspan(0, stim_dur, color=col, alpha=0.2)
            ax_avg.plot(t_win, avg, color=col, label=f"{freq} Hz")
            ax_avg.fill_between(t_win, avg-std, avg+std, color=col, alpha=0.3)
            ax_avg.set_xlabel("Temps relatif à la stim (s)")
            ax_avg.set_ylabel("Signal norm.")
            ax_avg.legend()
        plt.tight_layout()
        fig_name = os.path.join(figures_path, "Raster_GCaMP.png")
        plt.savefig(fig_name, dpi=300)
        plt.close(fig)
        print(f"Figure sauvegardée dans : {fig_name}")

# --- Hb ---
if DO_HB:
    hb_file = os.path.join(data_path, "computedHb.npy")
    hb_ts_file = os.path.join(data_path, "625ts.npy")
    if os.path.exists(hb_file) and os.path.exists(hb_ts_file):
        data = np.load(hb_file)  # (3, n_frames, y, x)
        ts = np.load(hb_ts_file)
        min_len = min(data.shape[1], len(ts))
        if data.shape[1] != len(ts):
            print(f"Attention: computedHb.npy ({data.shape[1]}) et 625ts.npy ({len(ts)}) n'ont pas le même nombre de frames. Tronquage à {min_len}.")
            data = data[:, :min_len, :, :]
            ts = ts[:min_len]
        modalities = ['HbO', 'HbR', 'HbT']
        modal_colors = ['crimson', 'dodgerblue', 'darkorange']
        for idx, (mod, mod_color) in enumerate(zip(modalities, modal_colors)):
            trace = data[idx].mean(axis=(1,2))
            trace_norm = (trace - trace.min()) / (trace.max() - trace.min())
            trial_sets = []
            t_windows = []
            for stim_times in stim_sets:
                trials, t_win = extract_trials(trace_norm, ts, stim_times, window_before, stim_dur, window_after)
                trial_sets.append(trials)
                t_windows.append(t_win)
            fig = plt.figure(figsize=(18, 10))
            fig.suptitle(f"Raster - {mod}")
            ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
            ax0.plot(ts, trace_norm, color=mod_color, label=mod)
            for stim, col, label in zip(stim_sets, colors, [f"{f} Hz" for f in freqs]):
                for s in stim:
                    ax0.axvline(s, color=col, linestyle='--', alpha=0.7, label=label if s == stim[0] else "")
            ax0.set_xlabel("Temps (s)")
            ax0.set_ylabel("Signal normalisé")
            handles, labels_ = ax0.get_legend_handles_labels()
            by_label = dict(zip(labels_, handles))
            ax0.legend(by_label.values(), by_label.keys())
            for i, (freq, trials, t_win, col) in enumerate(zip(freqs, trial_sets, t_windows, colors)):
                ax_raster = plt.subplot2grid((3, 3), (1, i))
                im = ax_raster.imshow(trials, aspect='auto', extent=[t_win[0], t_win[-1], 0, trials.shape[0]],
                                      origin='lower', cmap='Greens', interpolation='none')
                ax_raster.axvspan(0, stim_dur, color=col, alpha=0.2)
                ax_raster.set_title(f"{freq} Hz")
                ax_raster.set_xlabel("Temps relatif à la stim (s)")
                ax_raster.set_ylabel("Essai")
                plt.colorbar(im, ax=ax_raster, label="Signal norm.")
                ax_avg = plt.subplot2grid((3, 3), (2, i))
                avg = trials.mean(axis=0)
                std = sem(trials, axis=0)
                ax_avg.axvspan(0, stim_dur, color=col, alpha=0.2)
                ax_avg.plot(t_win, avg, color=col, label=f"{freq} Hz")
                ax_avg.fill_between(t_win, avg-std, avg+std, color=col, alpha=0.3)
                ax_avg.set_xlabel("Temps relatif à la stim (s)")
                ax_avg.set_ylabel("Signal norm.")
                ax_avg.legend()
            plt.tight_layout()
            fig_name = os.path.join(figures_path, f"Raster_{mod}.png")
            plt.savefig(fig_name, dpi=300)
            plt.close(fig)
            print(f"Figure sauvegardée dans : {fig_name}")

# --- LSCI ---
if DO_LSCI:
    lsci_file = os.path.join(data_path, "computedLSCI.npy")
    lsci_ts_file = os.path.join(data_path, "785ts.npy")
    if os.path.exists(lsci_file) and os.path.exists(lsci_ts_file):
        data = np.load(lsci_file)  # (n_frames, y, x)
        ts = np.load(lsci_ts_file)
        min_len = min(data.shape[0], len(ts))
        if data.shape[0] != len(ts):
            print(f"Attention: computedLSCI.npy ({data.shape[0]}) et 785ts.npy ({len(ts)}) n'ont pas le même nombre de frames. Tronquage à {min_len}.")
            data = data[:min_len]
            ts = ts[:min_len]
        trace = data.mean(axis=(1,2))
        trace_norm = (trace - trace.min()) / (trace.max() - trace.min())
        trial_sets = []
        t_windows = []
        for stim_times in stim_sets:
            trials, t_win = extract_trials(trace_norm, ts, stim_times, window_before, stim_dur, window_after)
            trial_sets.append(trials)
            t_windows.append(t_win)
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("Raster - LSCI")
        ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
        ax0.plot(ts, trace_norm, color='purple', label='LSCI')
        for stim, col, label in zip(stim_sets, colors, [f"{f} Hz" for f in freqs]):
            for s in stim:
                ax0.axvline(s, color=col, linestyle='--', alpha=0.7, label=label if s == stim[0] else "")
        ax0.set_xlabel("Temps (s)")
        ax0.set_ylabel("Signal normalisé")
        handles, labels_ = ax0.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        ax0.legend(by_label.values(), by_label.keys())
        for i, (freq, trials, t_win, col) in enumerate(zip(freqs, trial_sets, t_windows, colors)):
            ax_raster = plt.subplot2grid((3, 3), (1, i))
            im = ax_raster.imshow(trials, aspect='auto', extent=[t_win[0], t_win[-1], 0, trials.shape[0]],
                                  origin='lower', cmap='Greens', interpolation='none')
            ax_raster.axvspan(0, stim_dur, color=col, alpha=0.2)
            ax_raster.set_title(f"{freq} Hz")
            ax_raster.set_xlabel("Temps relatif à la stim (s)")
            ax_raster.set_ylabel("Essai")
            plt.colorbar(im, ax=ax_raster, label="Signal norm.")
            ax_avg = plt.subplot2grid((3, 3), (2, i))
            avg = trials.mean(axis=0)
            std = sem(trials, axis=0)
            ax_avg.axvspan(0, stim_dur, color=col, alpha=0.2)
            ax_avg.plot(t_win, avg, color=col, label=f"{freq} Hz")
            ax_avg.fill_between(t_win, avg-std, avg+std, color=col, alpha=0.3)
            ax_avg.set_xlabel("Temps relatif à la stim (s)")
            ax_avg.set_ylabel("Signal norm.")
            ax_avg.legend()
        plt.tight_layout()
        fig_name = os.path.join(figures_path, "Raster_LSCI.png")
        plt.savefig(fig_name, dpi=300)
        plt.close(fig)
        print(f"Figure sauvegardée dans : {fig_name}")

# --- Face motion ---
if DO_FACE:
    face_file = os.path.join(data_path, "face_motion.npy")
    if os.path.exists(face_file):
        face_motion = np.load(face_file)  # (n_frames_total,)
        ts_files = ["470ts.npy", "530ts.npy", "625ts.npy", "785ts.npy", "405ts.npy"]
        ts_list = []
        for f in ts_files:
            f_path = os.path.join(data_path, f)
            if os.path.exists(f_path):
                ts_list.append(np.load(f_path))
            else:
                print(f"Attention : {f} manquant, ignoré pour face_motion.")
        ts_concat = np.concatenate(ts_list)
        # --- On trie les timestamps et le signal associé ---
        sort_idx = np.argsort(ts_concat)
        ts_sorted = ts_concat[sort_idx]
        face_motion_sorted = face_motion[sort_idx]
        min_len = min(len(face_motion_sorted), len(ts_sorted))
        if len(face_motion_sorted) != len(ts_sorted):
            print(f"Attention: face_motion.npy ({len(face_motion_sorted)}) et concaténation des timestamps ({len(ts_sorted)}) n'ont pas le même nombre de frames. Tronquage à {min_len}.")
        face_motion_trunc = face_motion_sorted[:min_len]
        ts_sorted_trunc = ts_sorted[:min_len]
        face_norm = (face_motion_trunc - np.min(face_motion_trunc)) / (np.max(face_motion_trunc) - np.min(face_motion_trunc))

        # Extraction des essais pour chaque fréquence
        trial_sets = []
        t_windows = []
        for stim_times in stim_sets:
            trials, t_win = extract_trials(face_norm, ts_sorted_trunc, stim_times, window_before, stim_dur, window_after)
            trial_sets.append(trials)
            t_windows.append(t_win)

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("Raster - Face motion")

        # Signal global tronqué
        ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
        ax0.plot(ts_sorted_trunc, face_norm, color='black', label='Face motion (norm.)')
        for stim, col, label in zip(stim_sets, colors, [f"{f} Hz" for f in freqs]):
            for s in stim:
                ax0.axvline(s, color=col, linestyle='--', alpha=0.7, label=label if s == stim[0] else "")
        ax0.set_xlabel("Temps (s)")
        ax0.set_ylabel("Mouvement facial (normalisé)")
        handles, labels_ = ax0.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        ax0.legend(by_label.values(), by_label.keys())

        # Raster et moyenne ± SEM pour chaque fréquence
        for i, (freq, trials, t_win, col) in enumerate(zip(freqs, trial_sets, t_windows, colors)):
            # Raster
            ax_raster = plt.subplot2grid((3, 3), (1, i))
            im = ax_raster.imshow(trials, aspect='auto', extent=[t_win[0], t_win[-1], 0, trials.shape[0]],
                                  origin='lower', cmap='Greens', interpolation='none')
            ax_raster.axvspan(0, stim_dur, color=col, alpha=0.2)
            ax_raster.set_title(f"{freq} Hz")
            ax_raster.set_xlabel("Temps relatif à la stim (s)")
            ax_raster.set_ylabel("Essai")
            plt.colorbar(im, ax=ax_raster, label="Signal norm.")

            # Moyenne ± SEM
            ax_avg = plt.subplot2grid((3, 3), (2, i))
            avg = trials.mean(axis=0)
            std = sem(trials, axis=0)
            ax_avg.axvspan(0, stim_dur, color=col, alpha=0.2)
            ax_avg.plot(t_win, avg, color=col, label=f"{freq} Hz")
            ax_avg.fill_between(t_win, avg-std, avg+std, color=col, alpha=0.3)
            ax_avg.set_xlabel("Temps relatif à la stim (s)")
            ax_avg.set_ylabel("Signal norm.")
            ax_avg.legend()

        plt.tight_layout()
        fig_name = os.path.join(figures_path, "Raster_FaceMotion.png")
        plt.savefig(fig_name, dpi=300)
        plt.close(fig)
        print(f"Figure sauvegardée dans : {fig_name}")

# --- Pupil ---


pupille = np.load("temp-05232025092936-0000_proc.npy", allow_pickle=True).item()
area_smooth = pupille['pupil'][0]['area_smooth']
diam_smooth = 2 * np.sqrt(area_smooth / np.pi)


len_diam = len(diam_smooth)
frame_duration = 1343.8658 / len_diam
timestamps = np.arange(0, len_diam * frame_duration, frame_duration)

if DO_PUPIL:
    diam_norm = diam_smooth
    ts_sorted_trunc = timestamps

    # Extraction des essais pour chaque fréquence
    trial_sets = []
    t_windows = []
    for stim_times in stim_sets:
        trials, t_win = extract_trials(diam_norm, ts_sorted_trunc, stim_times, window_before, stim_dur, window_after)
        trial_sets.append(trials)
        t_windows.append(t_win)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Raster - Pupille")

    # Signal global tronqué
    ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    ax0.plot(ts_sorted_trunc, diam_norm, color='black', label='Diamètre pupille (norm.)')
    for stim, col, label in zip(stim_sets, colors, [f"{f} Hz" for f in freqs]):
        for s in stim:
            ax0.axvline(s, color=col, linestyle='--', alpha=0.7, label=label if s == stim[0] else "")
    ax0.set_xlabel("Temps (s)")
    ax0.set_ylabel("Diamètre pupille (normalisée)")
    handles, labels_ = ax0.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax0.legend(by_label.values(), by_label.keys())

    # Raster et moyenne ± SEM pour chaque fréquence
    for i, (freq, trials, t_win, col) in enumerate(zip(freqs, trial_sets, t_windows, colors)):
        # Raster
        ax_raster = plt.subplot2grid((3, 3), (1, i))
        im = ax_raster.imshow(trials, aspect='auto', extent=[t_win[0], t_win[-1], 0, trials.shape[0]],
                              origin='lower', cmap='Greens', interpolation='none')
        ax_raster.axvspan(0, stim_dur, color=col, alpha=0.2)
        ax_raster.set_title(f"{freq} Hz")
        ax_raster.set_xlabel("Temps relatif à la stim (s)")
        ax_raster.set_ylabel("Essai")
        plt.colorbar(im, ax=ax_raster, label="Diamètre norm.")

        # Moyenne ± SEM
        ax_avg = plt.subplot2grid((3, 3), (2, i))
        avg = trials.mean(axis=0)
        std = sem(trials, axis=0)
        ax_avg.axvspan(0, stim_dur, color=col, alpha=0.2)
        ax_avg.plot(t_win, avg, color=col, label=f"{freq} Hz")
        ax_avg.fill_between(t_win, avg-std, avg+std, color=col, alpha=0.3)
        ax_avg.set_xlabel("Temps relatif à la stim (s)")
        ax_avg.set_ylabel("Diamètre norm.")
        ax_avg.legend()

    plt.tight_layout()
    fig_name = os.path.join(figures_path, "Raster_Pupil.png")
    plt.savefig(fig_name, dpi=300)
    plt.close(fig)
    print(f"Figure sauvegardée dans : {fig_name}")
