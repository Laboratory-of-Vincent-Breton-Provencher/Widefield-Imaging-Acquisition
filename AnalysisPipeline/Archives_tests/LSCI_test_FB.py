import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *

# Sélection du fichier .npy
root = Tk()
root.withdraw()
npy_path = filedialog.askopenfilename(title="Sélectionner le fichier computedLSCI.npy", filetypes=[("Numpy files", "*.npy")])

# Chargement du stack
data = np.load(npy_path)
print("Stack shape:", data.shape)

# Moyenne temporelle
mean_img = np.mean(data, axis=0)

# Affichage
plt.figure(figsize=(6, 6))
plt.imshow(mean_img, cmap='hot')
plt.title("Image LSCI moyennée")
plt.colorbar(label="Flux relatif")
plt.axis('off')
plt.show()
