import numpy as np
import matplotlib.pyplot as plt
import re

def plot_all_channels(npy_files):
    plt.figure(figsize=(12, 6))
    for npy_file in npy_files:
        ts = np.load(npy_file)
        # Déterminer l'offset de départ selon l'ordre dans la liste (1 pour le premier, 2 pour le second, etc.)
        channel_index = npy_files.index(npy_file)
        start = channel_index + 1
        # Extraire le label du nom du fichier
        match = re.search(r'(\d+)', npy_file)
        label = match.group(1) + " nm" if match else npy_file
        # Indices globaux pour ce channel
        indices = np.arange(start, start + len(ts)*5, 5)
        plt.plot(indices, ts, marker='o', label=label)
    plt.title("Timestamps par channel")
    plt.xlabel("Numéro d'image global")
    plt.ylabel("Timestamp (s)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Exemple d'utilisation :
npy_files = [
    "Fichiers_ts_npy/405ts.npy",
    "Fichiers_ts_npy/470ts.npy",
    "Fichiers_ts_npy/530ts.npy",
    "Fichiers_ts_npy/625ts.npy",
    "Fichiers_ts_npy/785ts.npy"
]
plot_all_channels(npy_files)