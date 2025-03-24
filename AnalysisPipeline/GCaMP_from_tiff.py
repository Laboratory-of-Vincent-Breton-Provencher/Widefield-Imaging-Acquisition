import os
from tkinter import filedialog
from tkinter import *
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from prepData import create_npy_stack, prepToCompute, resample_pixel_value, save_as_tiff, create_list_trials



def GCaMP_pipeline(data_path:str, save_path:str, event_timestamps:list=None, preprocess:bool=True, nFrames:int=None,  correct_motion:bool=True, bin_size:int=2, regress:bool=False, filter_sigma:float=2):
    """ Analysis pipeline to process raw frames into neuronal activity GCaMP. Saves processed tiff files as well as
        the numpy 3D array containing the data.

    Args:
        data_path (str): path of the raw data
        save_path (str): path of where to save processed GCaMP data
        preprocess (bool, optional): Use False if preprocessed file already saved. Defaults to True.
        nFrames (int, optional): number of frames to analyse. If None, analyze all frames
        correct_motion (bool, optional): Corrects motion in images with antspy registration. Defaults to True.
        bin_size (int, optional): bins data to make it smaller. Defaults to 2.
        regress (bool, optional): normalizes the data around 1. Defaults to False.
        filter_sigma (float, optional): gaussian filter. None means no filter, otherwise specify sigma. Defaults to 2.
    """
    # Analyse du data complet sans trier essais. Attention à ne pas buster la ram (limiter avec nFrames)
    if event_timestamps is None:
        print("Analysis not by trial")
        if preprocess:
            # process blue
            print("Loading blue data")
            blue = create_npy_stack(data_path + "\\470", data_path, 470, saving=False, nFrames=nFrames)
            blue = prepToCompute(blue, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
            np.save(data_path + "\\470_preprocessed.npy", blue)
            blue = None
            print("Blue data preprocessed and saved")

            # process purple
            print("Loading purple data")
            purple = create_npy_stack(data_path + "\\405", data_path, 405, saving=False, nFrames=nFrames)
            purple = prepToCompute(purple, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
            np.save(data_path + "\\405_preprocessed.npy", purple)
            purple = None
            print("Purple data preprocessed and saved")

        # convert to neuronal activity
        print("Converting to neuronal activity")
        blue = np.load(data_path + "\\470_preprocessed.npy")
        purple = np.load(data_path + "\\405_preprocessed.npy")

        d_gcamp = zscore(blue-purple, axis=0)                       #######
        print(d_gcamp.shape)

        # resample pixel values
        d_gcamp = resample_pixel_value(d_gcamp, 16).astype(np.uint16)
        # filter if needed
        if filter_sigma is not None:
            print("Filtering")
            d_gcamp = gaussian_filter(d_gcamp, sigma=filter_sigma, axes=(0))
        # save processed data as npy
        np.save(save_path + "\\computedGCaMP.npy", d_gcamp)
        # save as tiff
        print("Saving processed GCaMP")

        test = "\\GCaMP_zscore_b-p"                                        #######
        
        try:
            os.mkdir(save_path + test)
        except FileExistsError:
            print("Folder already created")
        save_as_tiff(d_gcamp, "GCaMP", save_path + test)

        print("Done")
#%% trial wise
    # Analyse des essais un à la fois (air puffs, optogen.) plus long, mais risque moins de buster la ram
    else:
        print("Analysis by trial")
        files_by_trial_b = create_list_trials(data_path, 470, event_timestamps)
        files_by_trial_p = create_list_trials(data_path, 405, event_timestamps)

        for trial_idx in range(len(files_by_trial_b)):       
            if preprocess:
                # process blue
                print("Loading blue data")
                blue = create_npy_stack(data_path + "\\470", data_path, 470, saving=False, cutAroundEvent=files_by_trial_b[trial_idx])
                blue = prepToCompute(blue, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
                np.save(data_path + "\\470_preprocessed.npy", blue)
                blue = None
                print("Blue data preprocessed and saved")

                # process purple
                print("Loading purple data")
                purple = create_npy_stack(data_path + "\\405", data_path, 405, saving=False, cutAroundEvent=files_by_trial_p[trial_idx])
                purple = prepToCompute(purple, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
                np.save(data_path + "\\405_preprocessed.npy", purple)
                purple = None
                print("Purple data preprocessed and saved")

            # convert to neuronal activity
            print("Converting to dHb")
            blue = np.load(data_path + "\\470_preprocessed.npy")
            purple = np.load(data_path + "\\405_preprocessed.npy")

            d_gcamp = None

            # resample pixel values
            d_gcamp = resample_pixel_value(d_gcamp, 16).astype(np.uint16)
            # filter if needed
            if filter_sigma is not None:
                print("Filtering")
                d_gcamp = gaussian_filter(d_gcamp, sigma=filter_sigma, axes=(0))
            # save processed data as npy
            np.save(save_path + "\\computedGCaMP_trial{}.npy".format(trial_idx), d_gcamp)
            # save as tiff
            print("Saving processed GCaMP")
            try:
                os.mkdir(save_path + "\\GCaMP")
            except FileExistsError:
                print("Folder already created")
            save_as_tiff(d_gcamp, "GCaMP" + "_trial{}_".format(trial_idx+1), save_path + "\\GCaMP")

            print("Done with trial {}".format(trial_idx))

        print("Done for real now")
#%%

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    data_path = filedialog.askdirectory()
    save_path = data_path

    # AP_times = np.array([  12.01,   35.2 ,   46.51])
    # ,   74.12,   91.14,  103.63,  114.48,
    # 132.14,  142.77,  169.61,  182.33,  197.83,  209.56,  223.5 ,
    # 239.35,  252.31,  263.77,  279.97,  297.53,  310.62,  323.38,
    # 335.92,  365.67,  383.93,  402.83,  417.51,  430.48,  440.9 ,
    # # 456.7 ,  468.25,  480.64])
    # opto_stims = np.arange(30, 1000, 32)


    # Analysis not by trial
    GCaMP_pipeline(data_path, save_path, preprocess=False, bin_size=None, nFrames=500)

    # Analysis by trial
    # GCaMP_pipeline(data_path, save_path, event_timestamps=AP_times, bin_size=None)

    # import matplotlib.pyplot as plt
    # blue = np.load(r"D:\ggermain\2025-03-21_opto_M914\1_whiskerpad\470_preprocessed.npy")
    # purple = np.load(r"D:\ggermain\2025-03-21_opto_M914\1_whiskerpad\405_preprocessed.npy")
    # plt.imshow(blue[0]-purple[0])
    # plt.show()