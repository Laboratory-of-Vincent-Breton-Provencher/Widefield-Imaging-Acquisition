# Code utilisé par Gabrielle Germain pour générer une figure qui montre l'effet de la correction du mouvement

import numpy as np
import ants
import matplotlib.pyplot as plt
from prepData import identify_files
import tifffile as tff


path = r"C:\Users\gabri\Desktop\data\2024-09-17"


## franmes: 47-53, 56-67,  462-483, 512-517 <- best

files_list = identify_files(path + "\\530", ".tif")
file1 = files_list[513]
file2 = files_list[518]

im1 = ants.from_numpy(tff.TiffFile(file1).asarray())
im2 = ants.from_numpy(tff.TiffFile(file2).asarray())


registration = ants.registration(im1, im2, type_of_transform="Affine")

im_corrected = ants.apply_transforms(im1, im2, transformlist=registration["fwdtransforms"]).numpy()


fig, axs = plt.subplots(1, 3, figsize=(12,4))


ax = plt.subplot(1, 3, 1)
ax.set_title("a)", loc='left')
ax.imshow(im1.numpy() - im2.numpy())
plt.axis('off')

ax = plt.subplot(1, 3, 2)
ax.set_title("b)", loc='left')
ax.imshow(im1.numpy()-im_corrected)
plt.axis('off')

ax = plt.subplot(1, 3, 3)
ax.set_title("c)", loc='left')
ax.imshow(im2.numpy()-im_corrected)
plt.axis('off')

plt.savefig("motion_correction_example.png", dpi=600)

plt.show()

# plt.imshow(im1.numpy() - im2.numpy())
# plt.show()

# plt.imshow(im1.numpy()-im_corrected)
# plt.show()

# plt.imshow(im2.numpy()-im_corrected)
# plt.show()