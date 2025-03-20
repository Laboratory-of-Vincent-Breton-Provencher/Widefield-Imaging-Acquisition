import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import tifffile as tff
from scipy.ndimage import shift
from scipy.optimize import curve_fit
from skimage.registration import phase_cross_correlation
import ants


#%%  --- TIMESERIES ANALYSIS ---

def create_list_trialsTS(data_path:str, wl:int, event_times:list, Ns_bef:int=3, Ns_aft:int=10, skip_last:bool=True):
    """Creates a list that contains time stamps cropped and sorted into trials

    Args:
        data_path (str): folder path for the try, not directly the channel, the folder that contains all the channels
        wl (int): wavelength, ideally the name of the folder that contains the files to analyze
        event_times (list): array or list of all the event time stamps, i.e. air puffs delivery or optogenetic stim
        Ns_bef (int, optional): number of seconds to keep before event. Defaults to 3.
        Ns_aft (int, optional): number of seconds to keep after event. Defaults to 10.
        skip_last (bool, optional): skips last trial if not enough time after. Defaults to True

    Returns:
        list: list of all the time stamps sorted for each trial. Not an array in case the last trial is shorter, but it's useful to change it to an array before use.
    """

    timestamps = np.load(data_path + "\\{}ts.npy".format(wl))
    max_time = timestamps[-1]
    last_event = np.argmin(np.abs((event_times - max_time)))
    event_times = event_times[0:last_event+1]

    AP_idx = []
    for ti in event_times:
        AP_idx.append(np.argmin(np.absolute(timestamps-ti)))

    if skip_last:
        AP_idx = AP_idx[:-1]

    Nf_bef = Ns_bef*10
    Nf_aft = Ns_aft*10

    sorted_ts_idx = []
    for idx in AP_idx:
        trial_idx = [idx-Nf_bef, idx+Nf_aft]
        sorted_ts_idx.append(trial_idx)

    ts_by_trial = []
    for idx, trial_idx in enumerate(sorted_ts_idx):
        idx_inf = trial_idx[0]
        idx_sup = trial_idx[1]
        ts_by_trial.append(timestamps[idx_inf:idx_sup])

    return ts_by_trial



def regress_drift2D(sig:list, time:list)-> list:
    """Prepares raw data to calculate HbO and HbR: removes 
        drift if any, and normalizes around 1

    Args:
        sig (list): 1D array containing signal
        time (list): 1D array containing time
        
    Returns:
        list: returns only the signal in a 1D array. Time is the same.
    """
    def droite(x, a, b):
        return a*x + b
    
    popt, pcov = curve_fit(droite, time, sig)
    pcov = None
    sig_r = sig/droite(time, *popt)

    return sig_r


def prepToComputeTS(sig:list, time:list, regress=True):
    """ Applies dfferent preprocessing algorithms and tools to data. Possibility to add more. 

    Args:
        sig (list): 1D array of signal (timeseries)
        time (list): 1D array of time associated with sig. Must be same length
        regress (bool, optional): linear regression for LED drift. Centers data around 1. Defaults to True.
        
    Returns:
        _type_: 1D array of sig
    """

    if regress:
        print("Regressing data")
        sig = regress_drift2D(sig, time)

    return sig


#%%  --- TIFF ANALYSIS ---

def identify_files(path, keywords):
    items = os.listdir(path)
    files = []
    for item in items:
        if all(keyword in item for keyword in keywords):
            files.append(item)
    files = [os.path.join(path, f) for f in files]
    files.sort(key=lambda x: os.path.getmtime(x))
    return files


def resample_pixel_value(data, bits):
    plage = 2**bits - 1
    return (plage * (data - np.min(data))/(np.max(data - np.min(data))))


def save_as_tiff(frames, data_type, save_path):
    """Helps save tiff images more easily

    Args:
        frames (array): 3D array of one type of data, ex HbO, HbR, or HbT
        data_type (str): type of data, for file prefix
        save_path (str): folder to save data
    """
    for idx, frame in tqdm(enumerate(frames)):
        im = Image.fromarray(frame, mode='I;16')
        im.save(save_path + "\\{}.tiff".format(data_type + str(idx)), "TIFF")


def create_list_trials(data_path:str, wl:int, event_times:list, Ns_bef:int=3, Ns_aft:int=10, skip_last:bool=False): 
    """ Creates a list that contains files cropped and sorted into trials

    Args:
        data_path (str): folder path for the try, not directly the channel, the folder that contains all the channels
        wl (int): wavelength, ideally the name of the folder that contains the files to analyze
        event_times (list): array or list of all the event time stamps, i.e. air puffs delivery or optogenetic stim
        Ns_bef (int, optional): number of seconds to keep before event. Defaults to 3.
        Ns_aft (int, optional): number of seconds to keep after event. Defaults to 10.
        skip_last (bool, optional): skips last trial if not enough time after. Defaults to False

    Returns:
        list: 2 dimensional list that contains the file paths to the frames of every trial
    """

    files_list = identify_files(data_path + "\\{}".format(wl), ".tif")
    frames_timestamps = np.load(data_path + "\\{}ts.npy".format(wl))
    n_frames = len(files_list)
    max_time = frames_timestamps[n_frames]
    last_event = np.argmin(np.abs((event_times - max_time)))
    event_times = event_times[0:last_event+1]

    AP_idx = []
    for ti in event_times:
        AP_idx.append(np.argmin(np.absolute(frames_timestamps-ti)))

    if skip_last:
        AP_idx = AP_idx[:-1]

    Nf_bef = Ns_bef*10
    Nf_aft = Ns_aft*10

    sorted_frames_idx = []
    for idx in AP_idx:
        trial_idx = [idx-Nf_bef, idx+Nf_aft]
        sorted_frames_idx.append(trial_idx)

    files_by_trial = []
    for idx, trial_idx in enumerate(sorted_frames_idx):
        idx_inf = trial_idx[0]
        idx_sup = trial_idx[1]
        files_by_trial.append(files_list[idx_inf:idx_sup])

    return files_by_trial


def create_npy_stack(folder_path:str, save_path:str,  wl:int, saving=False, nFrames:int=None, cutAroundEvent=None):
    """creates a 3D npy stack of raw tiff images

    Args:
        folder_path (str): folder containing tiff frames
        save_path (str): folder to save npy stack
        wl (int): wavelength for saved file name
        nFrames (int, optional): number of frames to analyse. If None, analyze all frames
    """

    if cutAroundEvent is not None:
        files = cutAroundEvent
    
    else:
        files = identify_files(folder_path, "tif")

    if nFrames is not None:
        files = files[:nFrames]
    for idx, file in tqdm(enumerate(files)):
        frame = tff.TiffFile(file).asarray()
        if idx == 0:
            num_frames = len(files)
            frame_shape = frame.shape
            stack_shape = (num_frames, frame_shape[0], frame_shape[1])
            _3d_stack = np.zeros(stack_shape, dtype=np.uint16)
        _3d_stack[idx,:,:] = frame

    if saving:
        np.save(save_path+"\\{}_rawStack.npy".format(wl), _3d_stack)
    return _3d_stack


def motion_correction(frames):
    """Applies motion correction based on a phase cross correlation

    Args:
        frames (_type_): 3D array of frames before correction

    Returns:
        _type_: 3D array of frames after correction
    """
    # fixed_frame = frames[0,:,:]
    # motion_corrected = np.zeros((frames.shape), dtype=np.float32)
    # for idx, frame in tqdm(enumerate(frames)):
    #     if idx == 0:
    #         motion_corrected[0,:,:] = frame
    #         continue
    #     shifted, error, diffphase = phase_cross_correlation(fixed_frame, frame, upsample_factor=10)
    #     corrected_image = shift(frame, shift=(shifted[0], shifted[1]), mode='reflect')
    #     motion_corrected[idx,:,:] = corrected_image
    
    # shifted, error, diffphase, corrected_image, fixed_frame = None, None, None, None, None
    # return motion_corrected

    fixed_frame = ants.from_numpy(frames[0,:,:])
    motion_corrected = np.zeros((frames.shape), dtype=np.float32)
    for idx, frame in tqdm(enumerate(frames)):
        if idx == 0:
            motion_corrected[0,:,:] = frame
            continue
        moving_frame = ants.from_numpy(frame)
        registration = ants.registration(fixed_frame, moving_frame,  type_of_transform="Affine")
        motion_corrected[idx,:,:] = ants.apply_transforms(fixed_frame, moving_frame, transformlist=registration["fwdtransforms"]).numpy()

    return motion_corrected

def bin_pixels(frames, bin_size=2):
    """Bins pixels with bin size

    Args:
        frames (array): 3D array of frames. 
        bin_size (int, optional): size of pixel bins. Defaults to 2.

    Returns:
        array: 3D array, stack of binned data
    """
    for idx, frame in tqdm(enumerate(frames)):
        if idx == 0:
            height, width = frame.shape[:2]
            binned_height = height // bin_size
            binned_width = width // bin_size
            binned_frames = np.zeros((len(frames), binned_height, binned_width), dtype=np.uint16)

        reshaped_frame = frame[:binned_height * bin_size, :binned_width * bin_size].reshape(binned_height, bin_size, binned_width, bin_size)

        binned_frame = np.sum(reshaped_frame, axis=(1, 3), dtype=np.float32)
        binned_frame = binned_frame / (bin_size**2)
        binned_frames[idx,:,:] = binned_frame

    height, width, binned_height, binned_width, reshaped_frame = None, None, None, None, None
    return binned_frames


def regress_drift(sig:list, time:list)-> list:
    """Prepares raw data to calculate HbO and HbR: removes 
        drift if any, and normalizes around 1

    Args:
        sig (list): 1D array containing signal
        time (list): 1D array containing time
        
    Returns:
        list: returns only the signal in a 1D array. Time is the same.
    """
    def droite(x, a, b):
        return a*x + b
    
    print("Global regression")
    popt, pcov = curve_fit(droite, time, sig)
    pcov = None
    sig_r = sig/droite(time, *popt)

    return sig_r


def prepToCompute(frames:list, correct_motion:bool=False, bin_size:int=None, regress:bool=False)->list:
    """ preprocesses raw frames before computing Hb

    Args:
        frames (list): numpy array of raw frames
        correct_motion (bool, optional): Corrects motion in images with phase cross-correlation. Defaults to True.. Defaults to False.
        bin_size (int, optional):  bins data to make it smaller. Defaults to None.
        regress (bool, optional): normalizes the data around 1. Defaults to False.

    Returns:
        list: numpy array of preprocessed frames
    """
    if correct_motion:
        print("Correcting motion")
        frames = motion_correction(frames)
    if bin_size is not None:
        print("Bining pixels")
        frames = bin_pixels(frames, bin_size=bin_size)
    if regress:
        print("Normalizing")
        frames = frames/np.mean(frames, axis=0)
    
    return frames

#%% Tests

if __name__ == "__main__":
    from tkinter import filedialog
    from tkinter import *

    root = Tk()
    root.withdraw()
    data_path = filedialog.askdirectory()

    AP_times = np.array([  12.01,   35.2 ,   46.51,   74.12,   91.14,  103.63,  114.48,                         
                        132.14,  142.77,  169.61,  182.33,  197.83,  209.56,  223.5 ,
                        239.35,  252.31,  263.77,  279.97,  297.53,  310.62,  323.38,
                        335.92,  365.67,  383.93,  402.83,  417.51,  430.48,  440.9 ,
                        456.7 ,  468.25,  480.64])

    ts_sorted = create_list_trialsTS(data_path, 530, AP_times, skip_last=True)
    print(np.array(ts_sorted))
