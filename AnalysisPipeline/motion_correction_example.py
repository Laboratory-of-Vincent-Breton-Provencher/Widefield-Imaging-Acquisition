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

plt.imshow(im1.numpy() - im2.numpy())
plt.show()


registration = ants.registration(im1, im2, type_of_transform="Affine")

im_corrected = ants.apply_transforms(im1, im2, transformlist=registration["fwdtransforms"]).numpy()

plt.imshow(im1.numpy()-im_corrected)
plt.show()

plt.imshow(im2.numpy()-im_corrected)
plt.show()