import os
from tqdm import tqdm
import os
from tkinter import filedialog
from tkinter import *
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from prepData import create_npy_stack, prepToCompute, resample_pixel_value, save_as_tiff

def convertToLSCI(raw_speckle_data:list, window_size:int=7):
    """_summary_

    Args:
        raw_speckle_data (list): _description_
        window_size (int, optional): _description_. Defaults to 7.

    Returns:
        list: _description_
    """
    num_frames, height, width = raw_speckle_data.shape
    contrast_data = np.zeros((num_frames, height, width), dtype=np.float32)

    for frame_idx in tqdm(range(num_frames)):
        frame_data = raw_speckle_data[frame_idx]

        # Calculate local mean and standard deviation for the current frame
        local_mean = uniform_filter(frame_data, size=window_size)
        local_variance = uniform_filter(frame_data**2, size=window_size) - local_mean**2
        local_std = np.sqrt(local_variance)
        
        # Calculate speckle contrast for the current frame
        contrast_data[frame_idx] = 1/(local_std / local_mean)**2
    
    return contrast_data


def LSCI_pipeline(data_path:str, save_path:str, preprocess:bool=True, nFrames:int=None, correct_motion:bool=True, bin_size:int=2, regress:bool=False, filter_sigma:tuple=(2, 1, 1), window_size:int=5):
    """_summary_

    Args:
        data_path (str): _description_
        save_path (str): _description_
        preprocess (bool, optional): _description_. Defaults to True.
        nFrames (int, optional): number of frames to analyse. If None, analyze all frames
        correct_motion (bool, optional): _description_. Defaults to True.
        bin_size (int, optional): _description_. Defaults to 2.
        regress (bool, optional): _description_. Defaults to True.
        filter_sigma (tuple, optional): sigma values of filter. Defaults to (2, 1, 1).
        window_size (int, optional): _description_. Defaults to 7.
    """

    if preprocess:
        # preprocess 785 nm data
        print("Loading data")
        data = create_npy_stack(data_path + "\\785", data_path, 785, saving=False, nFrames=nFrames)
        data = data.astype(np.float32)
        print("Data loaded")
        data = prepToCompute(data, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
        np.save(data_path + "\\785_preprocessed.npy", data)
        data = None
        print("data preprocessed and saved")

    # converting to LSCI
    print("Converting to LSCI")
    data = np.load(data_path + "\\785_preprocessed.npy")
    data = data.astype(np.float32)
    data = convertToLSCI(data, window_size=window_size)
    if filter_sigma is not None:
        data = gaussian_filter(data, sigma=filter_sigma)
    data = resample_pixel_value(data, 16).astype(np.uint16)
    np.save(save_path + "\\computedLSCI.npy", data)
    print("Saving LSCI data")
    try:
        os.mkdir(save_path + "\\LSCI")
    except FileExistsError:
        print("Folder already created")
    save_as_tiff(data, "LSCI", save_path + "\\LSCI")

    print("Done")


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    data_path = filedialog.askdirectory()
    save_path = data_path
    nFrames = 200
    LSCI_pipeline(data_path, save_path, preprocess=False, nFrames=nFrames)