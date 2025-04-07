import os
from tkinter import filedialog, simpledialog
from tkinter import *
import tkinter as tk
import shutil
from tqdm import tqdm
import numpy as np
import tifffile

#%% Functions
def find_fname(Path:str, extension:str) -> list:
    """Retrieves files with a specific extension in a chosen folder

    Args:
        Path (str):
        extension (str):

    Returns:
        list: all the files with the chosen extension (not complete path, only file name)
    """
    ls_files = [file for file in os.listdir(Path) if os.path.isfile(os.path.join(Path, file))]
    fname = [s for s in ls_files if extension in s]
    # if len(fname) > 1:
    #     print('There is more than one {} file in {}'.format(contains,os.path.sep+P))
    #     sys.exit()
    
    return fname
#%%

## Paramètres: quoi faire et avec quels canaux

splitChannels = 0
extractTs = 1

FLAG405 = 0
FLAG470 = 1
FLAG530 = 1
FLAG625 = 1
FLAG785 = 1


FLAGS = {"FLAG405":FLAG405, "FLAG470":FLAG470, "FLAG530":FLAG530, "FLAG625":FLAG625, "FLAG785":FLAG785}

#%%  Channels Selection Window

# window = tk.Tk()
# window.withdraw()

# class ChannelSelectionWindow(simpledialog.Dialog):
#     def __init__(self, parent, title=None):
#         self.result = None
#         super().__init__(parent, title=title)

#     def body(self, frame):
#         tk.Label(frame, text='Select active channels', height=2, font=("Arial", 12)).grid(row=0)

#         self.var405 = tk.IntVar(value=1)
#         self.c405 = tk.Checkbutton(frame, text='405 nm (GCaMP isosbestic)',variable=self.var405, onvalue=1, offvalue=0, ).grid(row=1, sticky='W')

#         self.var470 = tk.IntVar(value=1)
#         self.c470 = tk.Checkbutton(frame, text='470 nm (GCaMP excitation)',variable=self.var470, onvalue=1, offvalue=0).grid(row=2, sticky='W')

#         self.var530 = tk.IntVar(value=1)
#         self.c530 = tk.Checkbutton(frame, text='530 nm (IOI isosbestic)',variable=self.var530, onvalue=1, offvalue=0).grid(row=3, sticky='W')

#         self.var625 = tk.IntVar(value=1)
#         self.c625 = tk.Checkbutton(frame, text='625 nm (IOI)',variable=self.var625, onvalue=1, offvalue=0).grid(row=4, sticky='W')

#         self.var785 = tk.IntVar(value=1)
#         self.c785 = tk.Checkbutton(frame, text='785 nm (LSCI)',variable=self.var785, onvalue=1, offvalue=0).grid(row=5, sticky='W')

#         # return self.var405, self.var470, self.var530, self.var625, self.var785

# test = ChannelSelectionWindow(window)

#%%

root = tk.Tk()
root.withdraw()
folderPath = filedialog.askdirectory()




if splitChannels:
    print("--- Split channels ---")
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


if extractTs:
    print("--- Extract time stamps ---")
    for flagName, flag in zip(FLAGS.keys(), FLAGS.values()):
        if flag:
            print("    Working on folder {}".format(flagName[4:7]))
            # Get list of fnames
            print('Getting file names')
            ls_f = np.array(find_fname(os.path.join(folderPath, flagName[4:7]),'.tif'))

            # Sort list of fnames
            print('Sorting files')
            idx_f = [int(f[10:f.find('.')]) for f in ls_f]
            idx_sort = np.argsort(idx_f)
            ls_f = ls_f[idx_sort]

            #%%
            # Run through the tiff stack to extract time stamps and ttl
            print('Extracting timestamps')
            ts = []
            for f in tqdm(ls_f):
                # print('Extracting timestamps for {}'.format(f))
                p = os.path.join(folderPath, flagName[4:7],f)

                with tifffile.TiffFile(p) as tif:
                    # Iterate over pages
                    for i, page in enumerate(tif.pages):
                
                        # Extract ttl for this frame (WORK IN PROGRESS)
                        x = page.description
                        i0 = x.find('bofTime')
                        a = x[i0:].find('=')
                        b = x[i0:].find('us')
                        ts.append(float(x[i0+a+1:i0+b]))

            # convert to sec
            ts = np.array(ts) * 10**-6
            ts -= ts[0]
            # delta_t = np.array(delta_t) * 10**-12

            # print(ts)
            # Save time stamps
            np.save(os.path.join(folderPath, flagName[4:7]) + 'ts.npy',ts)
            print('Timestamps file was created for folder {}'.format(flagName[4:7]))


#%% if GUI
# if __name__ == '__main__':
#     window.mainloop()