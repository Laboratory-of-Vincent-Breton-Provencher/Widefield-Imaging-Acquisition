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
folderList = ['530', '625']

filesList = []
for folder in folderList:
    folderFiles = os.listdir(os.path.join(folderPath, folder))
    for idx, file in enumerate(folderFiles):
        folderFiles[idx] = os.path.join(folderPath, folder, file)
    folderFiles.sort(key=lambda x: os.path.getmtime(x))
    filesList.append(folderFiles)

filesList530 = filesList[0]
filesList625 = filesList[1]


image = cv2.imread(filesList530[0], -1)

greenArray = np.zeros((np.shape(image)[0], np.shape(image)[1], len(filesList530)), dtype=np.uint16)
redArray = np.zeros((np.shape(image)[0], np.shape(image)[1], len(filesList625)), dtype=np.uint16)


print("Creating 3D data arrays")

for idx, file in enumerate(tqdm(filesList530)):
    image = cv2.imread(file, -1)
    greenArray[:,:,idx] = image

for idx, file in enumerate(tqdm(filesList625)):
    image = cv2.imread(file, -1)
    redArray[:,:,idx] = image


print("Analyzing data")

n_greenArray = aF.normalizeData(greenArray)
n_redArray =aF.normalizeData(redArray)
HbR, HbO = aF.oxygenation(n_greenArray, n_redArray)

print("Analysis completed")


print('Creating and saving images')

for idx in tqdm(range(1, np.shape(HbR)[2])):
    # HbR
    im = Image.fromarray(HbR[:,:,idx])
    im.save(fr"C:\Users\gabri\Desktop\testAnalyse\HbR_{idx}.tiff", "TIFF")
    # im.save("HbR_{}.tiff".format(idx), "TIFF")

    # HbO
    im = Image.fromarray(HbO[:,:,idx])
    im.save(fr"C:\Users\gabri\Desktop\testAnalyse\HbO_{idx}.tiff", "TIFF")
    # im.save("HbO_{}.tiff".format(idx), "TIFF")

