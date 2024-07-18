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

# folderPath = r"Y:\gGermain\2024-07-12\2 - Copie"
folderPath = r"C:\Users\gabri\Desktop\testAnalyse"
folderList = ['405', '470']

filesList = []
for folder in folderList:
    folderFiles = os.listdir(os.path.join(folderPath, folder))
    for idx, file in enumerate(folderFiles):
        folderFiles[idx] = os.path.join(folderPath, folder, file)
    folderFiles.sort(key=lambda x: os.path.getmtime(x))
    filesList.append(folderFiles)

filesList405 = filesList[0]
filesList470 = filesList[1]

image = cv2.imread(filesList470[0], -1)

violetArray = np.zeros((np.shape(image)[0], np.shape(image)[1], len(filesList405)), dtype=np.uint16)
blueArray = np.zeros((np.shape(image)[0], np.shape(image)[1], len(filesList470)), dtype=np.uint16)


print("Creating 3D data arrays")

for idx, file in enumerate(tqdm(filesList405)):
    image = cv2.imread(file, -1)
    violetArray[:,:,idx] = image

for idx, file in enumerate(tqdm(filesList470)):
    image = cv2.imread(file, -1)
    blueArray[:,:,idx] = image


print("Analyzing data")

n_blueArray = aF.normalizeData(blueArray)
n_violetArray =aF.normalizeData(violetArray)
delFF = aF.deltaFoverF(n_blueArray)
neuronalS = aF.hemodynamicCorrection(n_blueArray, n_violetArray)

print("Analysis completed")


print('Creating and saving images')

for idx in tqdm(range(1, np.shape(neuronalS)[2])):
    im = Image.fromarray(neuronalS[:,:,idx])
    im.save(fr"C:\Users\gabri\Desktop\testAnalyse\GCaMP_{idx}.tiff", "TIFF")
    # im.save("GCaMP_{}.tiff".format(idx), "TIFF")