import os
from tkinter import filedialog
from tkinter import *
import numpy as np
from scipy.ndimage import gaussian_filter
from ioiMatrices import ioi_epsilon_pathlength
from prepData import create_npy_stack, prepToCompute, resample_pixel_value, save_as_tiff, create_list_trials


def convertToHb(data_green:list, data_red:list):
    """converts green and red signals to Hb variation in tissue

    Args:
        data_green (list): array of preprocessed green frames
        data_red (list): array of preprocessed red frames

    Returns:
        tuple: 4D array of (d_HbO, d_HbR)
    """
    lambda1 = 450 #nm
    lamba2 = 700 #nm
    npoints = 1000
    baseline_hbt = 100 #uM
    baseline_hbo = 60 #uM
    baseline_hbr = 40 #uM
    rescaling_factor = 1e6
    
    eps_pathlength = ioi_epsilon_pathlength(lambda1, lamba2, npoints, baseline_hbt, baseline_hbo, baseline_hbr, filter=None)
    Ainv = (np.linalg.pinv(eps_pathlength)*rescaling_factor).astype(np.float32)
    ln_green = -np.log(data_green.flatten())
    ln_red = -np.log(data_red.flatten())
    ln_R = np.concatenate((ln_green.reshape(1,len(ln_green)),ln_red.reshape(1,len(ln_green))))
    Hbs = np.matmul(Ainv, ln_R).astype(np.float32)
    d_HbO = Hbs[0].reshape(np.shape(data_green))
    d_HbR = Hbs[1].reshape(np.shape(data_green))
    # Protection against aberrant data points
    np.nan_to_num(d_HbO, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(d_HbR, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    
    return d_HbO, d_HbR


def dHb_pipeline(data_path:str, save_path:str, event_timestamps:list=None, Ns_aft:int=10, preprocess:bool=True, nFrames:int=None,  correct_motion:bool=True, bin_size:int=2, regress:bool=True, filter_sigma:float=2):
    """ Analysis pipeline to process raw frames into Hb data. Saves processed tiff files as well as
        the numpy 4D array containing the 3 types of data (HbO, HbR, HbT).

    Args:
        data_path (str): path of the raw data
        save_path (str): path of where to save processed Hb data
        event_timestamps (list, optional): liste of event timestamps (air puffs, optogenetics). Defaults to None
        Ns_aft (int, optional): when processing by trial, how many seconds to analyse after event. Defaults to 10
        preprocess (bool, optional): Use False if preprocessed file already saved. Defaults to True.
        nFrames (int, optional): number of frames to analyse. If None, analyze all frames
        correct_motion (bool, optional): Corrects motion in images with antspy registration. Defaults to True.
        bin_size (int, optional): bins data to make it smaller. Defaults to 2.
        regress (bool, optional): normalizes the data around 1. Defaults to True.
        filter_sigma (float, optional): gaussian filter. None means no filter, otherwise specify sigma. Defaults to 2.5.
    """
    # Analyse du data complet sans trier essais. Attention à ne pas buster la ram (limiter avec nFrames)
    if event_timestamps is None:
        print("Analysis not by trial")
        if preprocess:
            # process green
            print("Loading green data")
            green = create_npy_stack(os.path.join(data_path, "530"), data_path, 530, saving=False, nFrames=nFrames)
            # green = np.load(data_path + "\\530_rawStack.npy")
            green = prepToCompute(green, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
            np.save(os.path.join(data_path, "530_preprocessed.npy"), green)
            green = None
            print("Green data preprocessed and saved")

            # process red
            print("Loading red data")
            red = create_npy_stack(os.path.join(data_path, "625"), data_path, 625, saving=False, nFrames=nFrames)
            # red = np.load(data_path + "\\625_rawStack.npy")
            red = prepToCompute(red, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
            np.save(os.path.join(data_path, "625_preprocessed.npy"), red)
            red = None
            print("Red data preprocessed and saved")

        # convert to hb
        print("Converting to dHb")
        green = np.load(os.path.join(data_path, "530_preprocessed.npy"))
        red = np.load(os.path.join(data_path, "625_preprocessed.npy"))
        d_HbO, d_HbR = convertToHb(green, red)
        d_HbT = d_HbO+d_HbR
        # resample pixel values
        d_HbO = resample_pixel_value(d_HbO, 16).astype(np.uint16)
        d_HbR = resample_pixel_value(d_HbR, 16).astype(np.uint16)
        d_HbT = resample_pixel_value(d_HbT, 16).astype(np.uint16)
        Hb = np.array((d_HbO, d_HbR, d_HbT))
        # filter if needed
        if filter_sigma is not None:
            print("Filtering")
            print(Hb.shape)
            Hb = gaussian_filter(Hb, sigma=filter_sigma, axes=(1))  # axe 1 parce 0 est le type de data
        # save processed data as npy
        np.save(os.path.join(data_path, "computedHb.npy"), Hb)
        # save as tiff
        print("Saving processed Hb")
        data_types = ['HbO', 'HbR', 'HbT']
        for frames, type in zip(Hb, data_types):
            try:
                os.mkdir(os.path.join(save_path, type))
            except FileExistsError:
                print("Folder already created")
            save_as_tiff(frames, type, os.path.join(save_path, type))

        print("Done")

    # Analyse des essais un à la fois (air puffs, optogen.) plus long, mais risque moins de buster la ram
    else:
        print("Analysis by trial")
        files_by_trial_g = create_list_trials(data_path, 530, event_timestamps, skip_last=True, Ns_aft=Ns_aft)
        files_by_trial_r = create_list_trials(data_path, 625, event_timestamps, skip_last=True, Ns_aft=Ns_aft)

        for trial_idx in range(len(files_by_trial_g)):
            if preprocess:
                # process green
                print("Loading green data")
                green = create_npy_stack(os.path.join(data_path, "530"), data_path, 530, saving=False, cutAroundEvent=files_by_trial_g[trial_idx])
                # green = np.load(data_path + "\\530_rawStack.npy")
                green = prepToCompute(green, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
                np.save(os.path.join(data_path, "530_preprocessed.npy"), green)
                green = None
                print("Green data preprocessed and saved")

                # process red
                print("Loading red data")
                red = create_npy_stack(os.path.join(data_path, 625), data_path, 625, saving=False, cutAroundEvent=files_by_trial_r[trial_idx])
                # red = np.load(data_path + "\\625_rawStack.npy")
                red = prepToCompute(red, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
                np.save(os.path.join(data_path, "625_preprocessed.npy"), red)
                red = None
                print("Red data preprocessed and saved")

            # convert to hb
            print("Converting to dHb")
            green = np.load(os.path.join(data_path, "530_preprocessed.npy"))
            red = np.load(os.path.join(data_path, "625_preprocessed.npy"))
            d_HbO, d_HbR = convertToHb(green, red)
            d_HbT = d_HbO+d_HbR
            # resample pixel values
            d_HbO = resample_pixel_value(d_HbO, 16).astype(np.uint16)
            d_HbR = resample_pixel_value(d_HbR, 16).astype(np.uint16)
            d_HbT = resample_pixel_value(d_HbT, 16).astype(np.uint16)
            Hb = np.array((d_HbO, d_HbR, d_HbT))
            # filter if needed
            if filter_sigma is not None:
                print("Filtering")
                Hb = gaussian_filter(Hb, sigma=filter_sigma, axes=(1))  # axe 1 parce 0 est le type de data
            # save processed data as npy
            try:
                os.mkdir(os.path.join(save_path, "computed_npy"))
            except FileExistsError:
                pass
            try:
                os.mkdir(os.path.join(save_path, "computed_npy", "Hb"))
            except FileExistsError:
                pass
            np.save(os.path.join(save_path, "computed_npy", "Hb", "computedHb_trial{}.npy".format(trial_idx+1)), Hb)
            # save as tiff
            print("Saving processed Hb")
            data_types = ['HbO', 'HbR', 'HbT']
            for frames, type in zip(Hb, data_types):
                try:
                    os.mkdir(os.path.join(save_path, type))
                except FileExistsError:
                    print("Folder already created")
                save_as_tiff(frames, type + "_trial{}_".format(trial_idx+1), os.path.join(save_path, type))

            print("----Done with trial {}".format(trial_idx+1))

        print("Done for real now")



if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    data_path = filedialog.askdirectory()
    save_path = data_path

    # Airpuffs
    #AP_times = np.load(r"AnalysisPipeline\Air_puff_timestamps.npy")
    
    # Optogénétique
    # attente = 30
    # stim = 5 #int(input("Duration of opto stim(to create adequate timestamps)"))
    # opto_stims = np.arange(attente, 1000, attente+stim)
    # Ns_aft = 15 #int(input("Seconds to analyze after onset of opto stim (trying to gte back to baseline)"))

    # Analysis not by trial
    # dHb_pipeline(data_path, save_path, preprocess=False, bin_size=None, nFrames=500)

    # Analysis by trial
    dHb_pipeline(data_path, save_path, event_timestamps=None, bin_size=2, nFrames=None, preprocess=True, filter_sigma=2)

