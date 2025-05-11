import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import RectangleSelector
from tkinter import filedialog
import os

# --- 1. Charger le fichier numpy
print("Sélectionne le fichier computedGCaMP.npy")
file_path = filedialog.askopenfilename(title="Sélectionne le fichier computedGCaMP.npy", filetypes=[("NumPy files", "*.npy")])
stack = np.load(file_path)
print("Stack shape:", stack.shape)

# --- 2. Affichage de l’animation GCaMP
fig_anim, ax_anim = plt.subplots()
frame_display = ax_anim.imshow(stack[0], cmap='viridis', vmin=stack.min(), vmax=stack.max())
frame_number = ax_anim.text(5, 5, "Frame 0", color='white', fontsize=12)
plt.title("Animation GCaMP (ferme cette fenêtre pour continuer)")

def update(frame):
    frame_display.set_data(stack[frame])
    frame_number.set_text(f"Frame {frame}")
    return [frame_display, frame_number]

ani = animation.FuncAnimation(fig_anim, update, frames=stack.shape[0], interval=50, blit=True)

# --- Option : sauvegarder la vidéo
save_video = True  # Mets True pour activer l’enregistrement

if save_video:
    output_path = os.path.splitext(file_path)[0] + "_gcamp.gif"
    print("Enregistrement de la vidéo (.gif)...")
    ani.save(output_path, writer='pillow', fps=10)
    print(f"✅ Vidéo sauvegardée dans : {output_path}")

# --- Affiche l’animation (bloque jusqu'à fermeture)
plt.show()

# --- 3. Sélection de ROI après l’animation
print("Dessine une ROI rectangulaire pour extraire un signal ΔF/F moyen")

roi = {}

def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    roi['x1'], roi['x2'] = sorted([x1, x2])
    roi['y1'], roi['y2'] = sorted([y1, y2])
    plt.close()

fig_roi, ax_roi = plt.subplots()
ax_roi.imshow(stack[0], cmap='viridis')
toggle_selector = RectangleSelector(ax_roi, onselect, useblit=True,
                                    button=[1], minspanx=5, minspany=5,
                                    spancoords='pixels', interactive=True)
plt.title("Dessine une ROI puis ferme la fenêtre")
plt.show()

# --- 4. Extraction du signal ROI
if roi:
    x1, x2 = roi['x1'], roi['x2']
    y1, y2 = roi['y1'], roi['y2']
    signal = np.mean(stack[:, y1:y2, x1:x2], axis=(1,2))

    fig_signal = plt.figure()
    plt.plot(signal)
    plt.title("Signal ΔF/F moyen dans la ROI")
    plt.xlabel("Frame")
    plt.ylabel("z-score")
    plt.grid()
    
    output_png = os.path.splitext(file_path)[0] + "_ROI_signal.png"
    fig_signal.savefig(output_png, dpi=300)
    print(f"✅ Graphique sauvegardé dans : {output_png}")
    plt.show()
else:
    print("Aucune ROI sélectionnée.")