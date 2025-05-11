import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

# Sélection du fichier computedHb.npy
file_path = filedialog.askopenfilename(title="Sélectionne le fichier computedHb.npy", filetypes=[("NumPy files", "*.npy")])
Hb = np.load(file_path)  # shape: (3, frames, height, width)
data_types = ['HbO', 'HbR', 'HbT']

# Moyenne temporelle
mean_images = Hb.mean(axis=1)  # shape: (3, height, width)

# Affichage côte à côte
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes):
    im = ax.imshow(mean_images[i], cmap='turbo')
    ax.set_title(data_types[i])
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Images moyennes (temporelles) pour HbO, HbR et HbT")
plt.tight_layout()
plt.show()
