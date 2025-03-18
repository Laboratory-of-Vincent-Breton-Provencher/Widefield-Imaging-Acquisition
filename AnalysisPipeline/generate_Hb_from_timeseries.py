import numpy as np
from ioiMatrices import ioi_epsilon_pathlength
from prepData import prepToComputeTS
from tkinter import filedialog
from tkinter import *
from scipy.ndimage import gaussian_filter

def convertToHbTS(data_green:list, data_red:list):
    """converts green and red signals to Hb variation in tissue

    Args:
        data_green (list): preprocessed green timeseries
        data_red (list): preprocessed red timeseries

    Returns:
        tuple: 2D array of (d_HbO, d_HbR) 
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


def dHb_pipeline(data_path:str, save_path:str, preprocess:bool=True, regress:bool=True, filter_sigma:float=2.5):
    """ Analysis pipeline to process raw timeseries into Hb data. Saves processed numpy array containing the 3 types of data (HbO, HbR, HbT).

    Args:
        data_path (string): path of the raw data. Data must be sorted first with the 'splitChannels.py' script, 
        and timestamps extracted with 'extract_ts_moment.py'
        save_path (string): path of folder where computed data must be saved
        preprocess (bool, optional): Use False if preprocessed file already saved. Defaults to True.
        regress (bool, optional): linear regression to remove LED drift and center data around 1. Defaults to True.
        filter_sigma (float, optional): gaussian filter. None means no filter, otherwise specify sigma. Defaults to 2.5
    """
    if preprocess:
        # process green
        green = np.loadtxt(data_path + "\\530.csv", skiprows=1, delimiter=',')[:,1]
        green_t = np.load(data_path + "\\530ts.npy")
        print("Green data loaded")
        green = prepToComputeTS(green, green_t, regress)
        np.save(data_path + "\\530_preprocessed.npy", green)
        green, green_t = None, None
        print("Green data saved")

        # process red    
        red = np.loadtxt(data_path + "\\625.csv", skiprows=1, delimiter=',')[:,1]
        red_t = np.load(data_path + "\\625ts.npy")
        print("Red data loaded")
        red = prepToComputeTS(red, red_t, regress)
        np.save(data_path + "\\625_preprocessed.npy", red)
        red = None
        print("Red data saved")

    # convert to hb
    print("Convert to dHb")
    green = np.load(data_path + "\\530_preprocessed.npy")
    red = np.load(data_path + "\\625_preprocessed.npy")
    d_HbO, d_HbR = convertToHbTS(green, red)
    Hb = np.array((d_HbO, d_HbR, d_HbO+d_HbR))

    if filter_sigma is not None:
        print("Filtering")
        Hb = gaussian_filter(Hb, sigma=filter_sigma, axes=(1)) # Vérifier axe

    # save processed data
    np.save(save_path + "\\computedHb.npy", Hb)
    print("Done")

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    data_path = filedialog.askdirectory()
    save_path = data_path
    dHb_pipeline(data_path, save_path, filter=True)