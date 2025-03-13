#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:05:18 2024

@author: vbp
"""

import numpy as np
import os
from tkinter import filedialog
from tkinter import *
from scipy.ndimage import gaussian_filter1d
import tifffile
from tqdm import tqdm

#%% FUNCTION

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
    
#%% PARAMS

# Path = r"Y:\gGermain\2024-07-18\4\785"
# Path = r"C:\Users\gabri\Desktop\testAnalyse\530"

root = Tk()
root.withdraw()
Path = filedialog.askdirectory()


#%% GET TIME STAMPS FROM ORIGINAL TIFF FILES

# Get list of fnames
print('Getting file names')
ls_f = np.array(find_fname(Path,'.tif'))

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
    p = os.path.join(Path,f)

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
np.save(Path + 'ts.npy',ts)
print('Timestamps file was created')