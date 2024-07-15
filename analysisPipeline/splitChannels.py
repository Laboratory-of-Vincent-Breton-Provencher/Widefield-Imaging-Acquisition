import os
from tkinter import filedialog
from tkinter import *
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import cmcrameri.cm as cmc
# cmap = 'cmc.batlow'
# import analysisFunctions as aF
from tqdm import tqdm
# from PIL import Image


FLAG405 = 1
FLAG470 = 1
FLAG530 = 1
FLAG625 = 1
FLAG785 = 1


# root = Tk()
# root.withdraw()
# folderPath = filedialog.askdirectory()
folderPath = r"C:\Users\gabri\Desktop\testAnalyse"

filesPaths = [os.path.join(folderPath, file) for file in os.listdir(folderPath)]
filesPaths.sort(key=lambda x: os.path.getmtime(x))
# print(filesPaths)


FLAGS = {"FLAG405":FLAG405, "FLAG470":FLAG470, "FLAG530":FLAG530, "FLAG630":FLAG625, "FLAG785":FLAG785}
# for flagName, flag in zip(FLAGS.keys(), FLAGS.values()):
#     # print(flagName, flag)
#     if flag:
#         os.makedirs(os.path.join(folderPath, flagName[4:7]))

# fin = 100

j = 0
filesSplitList = []
for flagName, flag in zip(FLAGS.keys(), FLAGS.values()):
    if flag:
        filesChannel = filesPaths[j::5]
        filesSplitList.append(filesChannel)
        j += 1
        print("filesChannel:")
        print(filesChannel)

print("filesSplitList:")
print(filesSplitList)


# filesList405 = 
# filesList470 = filesPaths[1:fin:5]
# filesList530 = filesPaths[2:fin:5]
# filesList625 = filesPaths[3:fin:5]
# filesList785 = filesPaths[4:fin:5]
