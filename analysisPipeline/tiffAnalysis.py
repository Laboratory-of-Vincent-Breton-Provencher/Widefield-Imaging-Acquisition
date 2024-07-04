import os
from tkinter import filedialog
from tkinter import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cmcrameri.cm as cmc
cmap = 'cmc.batlow'
import analysisFunctions as aF
from tqdm import tqdm
from PIL import Image


# root = Tk()
# root.withdraw()
# folder = filedialog.askdirectory()

folderPath = r"Z:\gGermain\28_JUIN_INTHEDARK"
filesList = os.listdir(folderPath)
filesPaths = []
for idx, file in enumerate(tqdm(filesList)):
    filesPaths.append(folderPath + "\\" + file)

filesPaths.sort(key=lambda x: os.path.getmtime(x))
filesPaths = filesPaths[1:]

filesListV = filesPaths[0:50:4]
filesListB = filesPaths[1:50:4]
# filesListG = filesPaths[2:50:4]
# filesListR = filesPaths[3:50:4]

# print(filesListB)

"""
FPS = 40
nChannels = 4
endTime = (1/FPS) * data.shape[0]
timestamp = np.linspace(0, int(endTime), int(data.shape[0]/nChannels))
"""

# print(filesListB[7])
image = cv2.imread(filesListB[0], -1)
# print(image)
# print(np.shape(image)[0])
blueArray = np.zeros((np.shape(image)[0], np.shape(image)[1], len(filesListB)))
violetArray = np.zeros((np.shape(image)[0], np.shape(image)[1], len(filesListV)))
# print(np.shape(blueArray[:,:,0]))
# fig, ax1 = plt.subplots()

for idx, file in enumerate(tqdm(filesListB)):
    image = cv2.imread(file, -1)
    blueArray[:,:,idx] = image

for idx, file in enumerate(tqdm(filesListV)):
    image = cv2.imread(file, -1)
    violetArray[:,:,idx] = image


n_blueArray = aF.normalizeData(blueArray)
n_violetArray =aF.normalizeData(violetArray)
delFF = aF.deltaFoverF(n_blueArray)
neuronalS = aF.hemodynamicCorrection(n_blueArray, n_violetArray)

# plt.plot(neuronalS[700, 700, :])
# plt.show()

for idx in range(np.shape(neuronalS)[2]):
    im = Image.fromarray(neuronalS[:,:,idx])
    # im.save("Z:\gGermain\28_JUIN_INTHEDARK\analyzed\GCaMP\GCaMP_{idx}.tiff", "TIFF")
    im.save("GCaMP_{}.tiff".format(idx), "TIFF")
    

# plt.imshow(blueArray[:,:,0], cmap=cmap)



