import os
from tkinter import filedialog
from tkinter import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cmcrameri.cm as cmc
cmap = 'cmc.batlow'
import analysisPipeline.Archives_tests.analysisFunctions as aF
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

fin = 100

filesListV = filesPaths[0:fin:4]
filesListB = filesPaths[1:fin:4]
filesListG = filesPaths[2:fin:4]
filesListR = filesPaths[3:fin:4]



image = cv2.imread(filesListB[0], -1)
# blueArray = np.zeros((np.shape(image)[0], np.shape(image)[1], len(filesListB)))
# violetArray = np.zeros((np.shape(image)[0], np.shape(image)[1], len(filesListV)))
redArray = np.zeros((np.shape(image)[0], np.shape(image)[1], len(filesListR)))
greenArray = np.zeros((np.shape(image)[0], np.shape(image)[1], len(filesListG)))



print("Creating 3D data arrays")

# for idx, file in enumerate(tqdm(filesListB)):
#     image = cv2.imread(file, -1)
#     blueArray[:,:,idx] = image

# for idx, file in enumerate(tqdm(filesListV)):
#     image = cv2.imread(file, -1)
#     violetArray[:,:,idx] = image

for idx, file in enumerate(tqdm(filesListR)):
    image = cv2.imread(file, -1)
    redArray[:,:,idx] = image

for idx, file in enumerate(tqdm(filesListG)):
    image = cv2.imread(file, -1)
    greenArray[:,:,idx] = image


# GCAMP

# n_blueArray = aF.normalizeData(blueArray)
# n_violetArray =aF.normalizeData(violetArray)
# delFF = aF.deltaFoverF(n_blueArray)
# neuronalS = aF.hemodynamicCorrection(n_blueArray, n_violetArray)


# IOI

print("Analyzing data")

n_greenArray = aF.normalizeData(greenArray)
n_redArray =aF.normalizeData(redArray)
HbR, HbO = aF.oxygenation(n_greenArray, n_redArray)

print("Analysis completed")

# print('Creating and saving images')

# for idx in tqdm(range(1, np.shape(neuronalS)[2])):
#     im = Image.fromarray(neuronalS[:,:,idx])
#     # im.save("Z:\gGermain\28_JUIN_INTHEDARK\analyzed\GCaMP\GCaMP_{idx}.tiff", "TIFF")
#     im.save("GCaMP_{}.tiff".format(idx), "TIFF")
#     # HbR
#     im = Image.fromarray(HbR[:,:,idx])
#     # im.save("Z:\gGermain\28_JUIN_INTHEDARK\analyzed\GCaMP\GCaMP_{idx}.tiff", "TIFF")
#     im.save("HbR_{}.tiff".format(idx), "TIFF")
#     # HbO
#     im = Image.fromarray(HbO[:,:,idx])
#     # im.save("Z:\gGermain\28_JUIN_INTHEDARK\analyzed\GCaMP\GCaMP_{idx}.tiff", "TIFF")
#     im.save("HbO_{}.tiff".format(idx), "TIFF")

