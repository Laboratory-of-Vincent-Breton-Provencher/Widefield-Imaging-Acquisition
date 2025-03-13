import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import tifffile as tff
from scipy.ndimage import shift
from scipy.optimize import curve_fit
from skimage.registration import phase_cross_correlation


#%%  --- TIMESERIES ANALYSIS ---

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


def save_as_tiff(frames, Hb, save_path):
    """Helps save tiff images more easily

    Args:
        frames (array): 3D array of one type of data, ex HbO, HbR, or HbT
        Hb (str): type of data, either HbO, HbR, or HbT
        save_path (str): folder to save data
    """
    for idx, frame in tqdm(enumerate(frames)):
        im = Image.fromarray(frame, mode='I;16')
        im.save(save_path + "\\{}.tiff".format(Hb + str(idx)), "TIFF")


def create_npy_stack(folder_path:str, save_path:str,  wl:int, saving=False):
    """creates a 3D npy stack of raw tiff images

    Args:
        folder_path (str): folder containing tiff frames
        save_path (str): folder to save npy stack
        wl (int): wavelength for saved file name
    """
    files = identify_files(folder_path, "tif")
    # files=files[:250]
    for idx, file in tqdm(enumerate(files)):
        # frame = tff.TiffFile(folder_path+"\\"+file).asarray()
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
    fixed_frame = frames[0,:,:]
    motion_corrected = np.zeros((frames.shape), dtype=np.uint16)
    for idx, frame in tqdm(enumerate(frames)):
        if idx == 0:
            motion_corrected[0,:,:] = frame
            continue
        shifted, error, diffphase = phase_cross_correlation(fixed_frame, frame, upsample_factor=10)
        corrected_image = shift(frame, shift=(shifted[0], shifted[1]), mode='reflect')
        motion_corrected[idx,:,:] = corrected_image
    
    shifted, error, diffphase, corrected_image, fixed_frame = None, None, None, None, None
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
