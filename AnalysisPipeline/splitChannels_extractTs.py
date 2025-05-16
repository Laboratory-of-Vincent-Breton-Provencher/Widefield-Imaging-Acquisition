import os
from tkinter import filedialog, simpledialog
from tkinter import *
import tkinter as tk
import shutil
from tqdm import tqdm
import numpy as np
import tifffile
import re  

#%% Functions
def find_fname(Path:str, *extensions:str) -> list:
    ls_files = [file for file in os.listdir(Path) if os.path.isfile(os.path.join(Path, file))]
    fname = [f for f in ls_files if f.lower().endswith(extensions)]
    return fname

def extract_number(filename):  # sort by numerical order
    match = re.search(r'\d+', os.path.basename(filename))
    return int(match.group()) if match else -1

#%%

splitChannels = 1
extractTs = 1

FLAG405 = 1
FLAG470 = 1
FLAG530 = 1
FLAG625 = 1
FLAG785 = 1

FLAGS = {"FLAG405":FLAG405, "FLAG470":FLAG470, "FLAG530":FLAG530, "FLAG625":FLAG625, "FLAG785":FLAG785}

#%%

root = tk.Tk()
root.withdraw()
folderPath = filedialog.askdirectory()

if splitChannels:
    print("--- Split channels ---")
    print("Sorting data")

    filesPaths = [os.path.join(folderPath, f) for f in os.listdir(folderPath) if f.lower().endswith(('.tif', '.tiff'))]
    filesPaths.sort(key=extract_number)

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

if extractTs:
    print("--- Extract time stamps ---")
    # 1. Chercher l'offset global (premier timestamp de tous les channels)
    all_first_ts = []
    for flagName, flag in zip(FLAGS.keys(), FLAGS.values()):
        if flag:
            ls_f = np.array(find_fname(os.path.join(folderPath, flagName[4:7]), '.tif', '.tiff'))
            if len(ls_f) == 0:
                continue
            idx_f = [extract_number(f) for f in ls_f]
            idx_sort = np.argsort(idx_f)
            ls_f = ls_f[idx_sort]
            # Prendre le premier fichier trié
            p = os.path.join(folderPath, flagName[4:7], ls_f[0])
            with tifffile.TiffFile(p) as tif:
                x = tif.pages[0].description
                i0 = x.find('bofTime')
                a = x[i0:].find('=')
                b = x[i0:].find('us')
                all_first_ts.append(float(x[i0+a+1:i0+b]))
    if len(all_first_ts) == 0:
        raise RuntimeError("Aucun timestamp trouvé dans les channels sélectionnés.")
    offset_global = min(all_first_ts)

    # 2. Extraction des timestamps pour chaque channel avec le même offset
    for flagName, flag in zip(FLAGS.keys(), FLAGS.values()):
        if flag:
            print("    Working on folder {}".format(flagName[4:7]))
            print('Getting file names')
            ls_f = np.array(find_fname(os.path.join(folderPath, flagName[4:7]), '.tif', '.tiff'))

            print('Sorting files')
            idx_f = [extract_number(f) for f in ls_f]
            idx_sort = np.argsort(idx_f)
            ls_f = ls_f[idx_sort]

            print('Extracting timestamps')
            ts = []
            for f in tqdm(ls_f):
                p = os.path.join(folderPath, flagName[4:7], f)
                with tifffile.TiffFile(p) as tif:
                    for i, page in enumerate(tif.pages):
                        x = page.description
                        i0 = x.find('bofTime')
                        a = x[i0:].find('=')
                        b = x[i0:].find('us')
                        ts.append(float(x[i0+a+1:i0+b]))

            ts = np.array(ts) * 1e-6
            ts -= offset_global * 1e-6  # On soustrait le même offset pour tous les channels

            np.save(os.path.join(folderPath, flagName[4:7]) + 'ts.npy', ts)
            print('Timestamps file was created for folder {}'.format(flagName[4:7]))

#%% if GUI
# if __name__ == '__main__':
#     window.mainloop()