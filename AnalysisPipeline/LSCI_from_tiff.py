import os
from tqdm import tqdm
from tkinter import filedialog
from tkinter import *
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from prepData import create_npy_stack, prepToCompute, resample_pixel_value, save_as_tiff, create_list_trials
import matplotlib.pyplot as plt

def convertToLSCI(raw_speckle_data:list, window_size:int=5):
    num_frames, height, width = raw_speckle_data.shape
    contrast_data = np.zeros((num_frames, height, width), dtype=np.float32)

    for frame_idx in tqdm(range(num_frames)):
        frame_data = raw_speckle_data[frame_idx]

        local_mean = uniform_filter(frame_data, size=window_size)
        local_variance = uniform_filter(frame_data**2, size=window_size) - local_mean**2
        local_std = np.sqrt(local_variance) + 1

        contrast_data[frame_idx] = 1/(local_std / local_mean)**2

    return contrast_data


def LSCI_pipeline(data_path:str, save_path:str, event_timestamps:list=None, Ns_aft:int=10, preprocess:bool=True, nFrames:int=None, correct_motion:bool=True, bin_size:int=2, regress:bool=False, filter_sigma:tuple=(2, 1, 1), window_size:int=5):
    if event_timestamps is None:
        if preprocess:
            print("Loading data")
            data = create_npy_stack(os.path.join(data_path, "785"), data_path, 785, saving=False, nFrames=nFrames)
            data = data.astype(np.float32)
            print("Data loaded")
            data = prepToCompute(data, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
            np.save(os.path.join(data_path, "785_preprocessed.npy"), data)
            data = None
            print("data preprocessed and saved")

        print("Converting to LSCI")
        data = np.load(os.path.join(data_path, "785_preprocessed.npy"))
        data = data.astype(np.float32)
        data = convertToLSCI(data, window_size=window_size)
        if filter_sigma is not None:
            data = gaussian_filter(data, sigma=filter_sigma)
        data = resample_pixel_value(data, 16).astype(np.uint16)
        np.save(os.path.join(save_path, "computedLSCI.npy"), data)
        print("Saving LSCI data")
        try:
            os.mkdir(os.path.join(save_path, "LSCI"))
        except FileExistsError:
            print("Folder already created")
        save_as_tiff(data, "LSCI", os.path.join(save_path, "LSCI"))

        print("Done")

    else:
        print("Analysis by trial")
        files_by_trial = create_list_trials(data_path, 785, event_timestamps, skip_last=True, Ns_aft=Ns_aft)

        for trial_idx in range(len(files_by_trial)):
            if preprocess:
                print("Loading data")
                data = create_npy_stack(os.path.join(data_path, "785"), data_path, 785, saving=False, cutAroundEvent=files_by_trial[trial_idx])
                data = data.astype(np.float32)
                print("Data loaded")
                data = prepToCompute(data, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
                np.save(os.path.join(data_path, "785_preprocessed.npy"), data)
                data = None
                print("data preprocessed and saved")

            print("Converting to LSCI")
            data = np.load(os.path.join(data_path, "785_preprocessed.npy"))
            data = data.astype(np.float32)
            data = convertToLSCI(data, window_size=window_size)

            if filter_sigma is not None:
                data = gaussian_filter(data, sigma=filter_sigma)

            data = resample_pixel_value(data, 16).astype(np.uint16)

            try:
                os.mkdir(os.path.join(save_path, "computed_npy"))
            except FileExistsError:
                pass
            try:
                os.mkdir(os.path.join(save_path, "computed_npy", "LSCI"))
            except FileExistsError:
                pass
            np.save(os.path.join(save_path, "computed_npy", "LSCI", f"computedLSCI_trial{trial_idx+1}.npy"), data)

            print("Saving processed LSCI")
            try:
                os.mkdir(os.path.join(save_path, "LSCI"))
            except FileExistsError:
                print("Folder already created")
            save_as_tiff(data, f"LSCI_trial{trial_idx+1}_", os.path.join(save_path, "LSCI"))

            print(f"----Done with trial {trial_idx+1}")

        print("Done for real now")


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    data_path = filedialog.askdirectory()
    save_path = data_path

    # Analysis not by trial
    LSCI_pipeline(data_path, save_path, event_timestamps=None, preprocess=True, nFrames=None, correct_motion=False, bin_size=2, regress=False, filter_sigma=(2, 1, 1), window_size=5)