import numpy as np
from scipy.signal import butter, filtfilt  # find_peaks, correlate, freqs
from scipy.ndimage import gaussian_filter # median_filter, shift
from scipy.optimize import curve_fit
from ioi_epsilon_pathlength_calc import ioi_epsilon_pathlength

from tkinter import filedialog
from tkinter import *


def lowpass_filter2D(sig:list, cutoff:float=1, fs:float=10, order:int=5)->list:
    """lowpass filter easy to use for data

    Args:
        sig (list): 1D array of data (timeseries, flattenned frames)
        cutoff (float): cutoff frequency.  Defaults to 1
        fs (float, optional): sampling frequency. Defaults to 10.
        order (int, optional): order of the butter filter. Defaults to 5.

    Returns:
        list: filtered data
    """
    nyq = 0.5 * fs
    low = cutoff / nyq
    b, a = butter(order, low, btype='low')
    filtered_data = filtfilt(b, a, sig, axis=0)
    return filtered_data
    

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


def prepToCompute2D(sig:list, time:list, filter=False, regress=True):
    """_summary_

    Args:
        sig (list): 1D array of signal (timeseries)
        time (list): 1D array of time associated with sig. Must be same length
        filter (bool, optional): lowpass filter. Defaults to False.
        cutoff (float, optional): cutoff frequency for lowpass filter. Defaults to 1.
        regress (bool, optional): linear regression for LED drift. Centers data around 1. Defaults to True.

    Returns:
        _type_: 1D array of sig
    """
    if filter:
        print("Filtering data")
        # sig = lowpass_filter2D(sig, cutoff)
        sig = gaussian_filter(sig, sigma=1)

    if regress:
        print("Regressing data")
        sig = regress_drift2D(sig, time)

    return sig

def convertToHb2D(data_green, data_red):
    """converts green and red signals to Hb variation in tissue

    Args:
        data_green (list): preprocessed green timeseries
        data_red (list): preprocessed red timeseries

    Returns:
        list: 2D array (d_HbO, d_HbR) 
    """
    lambda1 = 450 #nm
    lamba2 = 700 #nm
    npoints = 1000
    baseline_hbt = 100 #uM
    baseline_hbo = 60 #uM
    baseline_hbr = 40 #uM
    rescaling_factor = 1e6
    
    eps_pathlength = ioi_epsilon_pathlength(lambda1, lamba2, npoints, baseline_hbt, baseline_hbo, baseline_hbr, filter=None)
    Ainv = np.linalg.pinv(eps_pathlength)*rescaling_factor
    ln_green = -np.log(data_green.flatten())
    ln_red = -np.log(data_red.flatten())
    ln_R = np.concatenate((ln_green.reshape(1,len(ln_green)),ln_red.reshape(1,len(ln_green))))
    Hbs = np.matmul(Ainv, ln_R)
    d_HbO = Hbs[0].reshape(np.shape(data_green))
    d_HbR = Hbs[1].reshape(np.shape(data_green))
    # Protection against aberrant data points
    np.nan_to_num(d_HbO, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(d_HbR, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    
    return d_HbO, d_HbR


def dHb_pipeline(data_path, save_path, filter=False, regress=True):
    """pipeline to compute Hb from raw data. Data must the sorted first, see data_path arg.

    Args:
        data_path (string): path of the raw data. Data must be sorted first with the 'splitChannels.py' script, 
        and timestamps extracted with 'extract_ts_moment.py'
        save_path (string): path of folder where computed data must be saved
        filter (bool, optional): lowpass time filter. Defaults to False.
        cutoff (float, optional): cutoff frequency if filter used. Defaults to 0.2.
        regress (bool, optional): linear regression to remove LED drift and center data around 1. Defaults to True.
    """
    # process green
    green = np.loadtxt(data_path + "\\530.csv", skiprows=1, delimiter=',')[:,1]
    green_t = np.load(data_path + "\\530ts.npy")
    print("Green data loaded")
    green = prepToCompute2D(green, green_t, filter, regress)
    np.save(data_path + "\\530preped.npy", green)
    green = None
    print("Green data saved")

    # process red    
    red = np.loadtxt(data_path + "\\625.csv", skiprows=1, delimiter=',')[:,1]
    red_t = np.load(data_path + "\\625ts.npy")
    print("Red data loaded")
    red = prepToCompute2D(red, red_t, filter, regress)
    np.save(data_path + "\\625preped.npy", red)
    red = None
    print("Red data saved")

    # convert to hb
    print("Convert to dHb")
    green = np.load(data_path + "\\530preped.npy")
    red = np.load(data_path + "\\625preped.npy")
    d_HbO, d_HbR = convertToHb2D(green, red)
    Hb = np.array((d_HbO, d_HbR, d_HbO+d_HbR))

    # save processed data
    np.save(save_path + "\\computedHb.npy", Hb)
    print("Done")

# test = False
# if test:
#     data_path = r"C:\Users\gabri\Desktop\testAnalyse\2024_07_18"
#     save_path = data_path
#     dHb_pipeline2D(data_path, save_path, filter=True, regress=True)


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    data_path = filedialog.askdirectory()
    save_path = data_path
    dHb_pipeline(data_path, save_path, filter=True)