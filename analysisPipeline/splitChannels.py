import os
from tkinter import filedialog
from tkinter import *
import shutil
from tqdm import tqdm


FLAG405 = 1
FLAG470 = 1
FLAG530 = 1
FLAG625 = 1
FLAG785 = 1

FLAGS = {"FLAG405":FLAG405, "FLAG470":FLAG470, "FLAG530":FLAG530, "FLAG625":FLAG625, "FLAG785":FLAG785}

root = Tk()
root.withdraw()
folderPath = filedialog.askdirectory()
# folderPath = r"C:\Users\gabri\Desktop\testAnalyse"


print("Sorting data")

filesPaths = [os.path.join(folderPath, file) for file in os.listdir(folderPath)]
filesPaths.sort(key=lambda x: os.path.getmtime(x))


j = 0
for flagName, flag in zip(FLAGS.keys(), FLAGS.values()):
    if flag:
        try:
            os.makedirs(os.path.join(folderPath, flagName[4:7]))
            print("Directory {} was created successfully".format(flagName[4:7]))
        except OSError as error:
            print("Directory {} already exists".format(flagName[4:7]))
        filesChannel = filesPaths[j::sum(FLAGS.values())]

        for file in tqdm(filesChannel):
            splitPath = os.path.split(file)
            shutil.move(file, os.path.join(splitPath[0], flagName[4:7], splitPath[1]))
        j += 1
