import os
from tqdm import tqdm
import os
from tkinter import filedialog
from tkinter import *
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from prepData import create_npy_stack, prepToCompute, resample_pixel_value, save_as_tiff, create_list_trials

def convertToLSCI(raw_speckle_data:list, window_size:int=5):
    """_summary_

    Args:
        raw_speckle_data (list): _description_
        window_size (int, optional): _description_. Defaults to 5.

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


def LSCI_pipeline(data_path:str, save_path:str, event_timestamps:list=None, Ns_aft:int=10, preprocess:bool=True, nFrames:int=None, correct_motion:bool=True, bin_size:int=2, regress:bool=False, filter_sigma:tuple=(2, 1, 1), window_size:int=5):
    """_summary_

    Args:
        data_path (str): _description_
        save_path (str): _description_
        event_timestamps (list, optional): liste of event timestamps (air puffs, optogenetics). Defaults to None
        Ns_aft (int, optional): when processing by trial, how many seconds to analyse after event. Defaults to 10
        preprocess (bool, optional): _description_. Defaults to True.
        nFrames (int, optional): number of frames to analyse. If None, analyze all frames
        correct_motion (bool, optional): _description_. Defaults to True.
        bin_size (int, optional): Use None if no bining is needed. Defaults to 2.
        regress (bool, optional): _description_. Defaults to True.
        filter_sigma (tuple, optional): sigma values of filter. Defaults to (2, 1, 1).
        window_size (int, optional): _description_. Defaults to 7.
    """

    # Analyse du data complet sans trier essais. Attention à ne pas buster la ram (limiter avec nFrames)
    if event_timestamps is None:
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

#%% trial wise
    # Analyse des essais un à la fois (air puffs, optogen.) plus long, mais risque moins de buster la ram
    else:
        print("Analysis by trial")
        files_by_trial = create_list_trials(data_path, 785, event_timestamps, skip_last=True, Ns_aft=Ns_aft)

        for trial_idx in range(len(files_by_trial)):       
            if preprocess:
                # process 785 nm
                print("Loading data")
                data = create_npy_stack(data_path + "\\785", data_path, 785, saving=False, cutAroundEvent=files_by_trial[trial_idx])
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

            # filter if needed
            if filter_sigma is not None:
                data = gaussian_filter(data, sigma=filter_sigma)
            
            # resample pixel value
            data = resample_pixel_value(data, 16).astype(np.uint16)

            # save processed data as npy
            try:
                os.mkdir(save_path + "\\computed_npy")
            except FileExistsError:
                pass
            try:
                os.mkdir(save_path + "\\computed_npy\\LSCI")
            except FileExistsError:
                pass
            np.save(save_path + "\\computed_npy\\LSCI\\computedLSCI_trial{}.npy".format(trial_idx+1), data)
            # save as tiff
            print("Saving processed Hb")
            try:
                os.mkdir(save_path + "\\LSCI")
            except FileExistsError:
                print("Folder already created")
            save_as_tiff(data, "LSCI" + "_trial{}_".format(trial_idx+1), save_path + "\\LSCI")

            print("----Done with trial {}".format(trial_idx+1))

        print("Done for real now")
#%%

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    data_path = filedialog.askdirectory()
    save_path = data_path

    AP_times = np.load(r"AnalysisPipeline\Air_puff_timestamps.npy")
    # attente = 30
    # stim = 5 #int(input("Duration of opto stim(to create adequate timestamps)"))
    # opto_stims = np.arange(attente, 1000, attente+stim)
    Ns_aft = 15 #int(input("Seconds to analyze after onset of opto stim (trying to gte back to baseline)"))

    # Analysis not by trial
    # LSCI_pipeline(data_path, save_path, preprocess=False, nFrames=500)

    # Analysis by trial
    LSCI_pipeline(data_path, save_path, AP_times, bin_size=None, Ns_aft=Ns_aft)