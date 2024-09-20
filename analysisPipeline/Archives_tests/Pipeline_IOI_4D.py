import os
from tqdm import tqdm

from tkinter import filedialog
from tkinter import *
from PIL import Image
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter, shift
from scipy.optimize import curve_fit

import tifffile as tff
from skimage.registration import phase_cross_correlation
import csv


def ioi_path_length_factor(lambda1, lambda2, npoints):
    """
    Return the pathlength values in cm from Ma. et al., Phil. Trans. R. Soc. B 371: 20150360.
    Values are stored in the 'Ma_values.txt' file.
    Parameters
    ----------
    lambda1/lambda2: scalar
        Wavelengths between which to return pathlength values
    npoints: int
        Number of sampling points between lambda1/2.
    Returns
    -------
    pathlengths: 1darray
        Pathlength values in mm between lambda1 and lambda2
    """
    with open(r"C:\Users\gabri\Documents\Université\Maitrise\Projet\Widefield-Imaging-Acquisition\analysisPipeline\Ma_values.txt", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        ma_values = []
        for row in reader:
            ma_values.append(list(map(float, row)))
    z = np.array(ma_values) #col1: wavelengths, col2: z in mm
    z[:,1] = z[:,1]/10 #Convert to cm
    if z[0,0] > lambda1:
        z = np.concatenate((np.array([[lambda1, 0], [z[0, 0]*0.9999, 0]]), z), axis=0)
    if z[-1,0] < lambda2:
        z = np.concatenate((z, np.array([[z[-1, 0]*1.00001, 0], [lambda2, 0]])), axis=0)
    xi = np.linspace(lambda1, lambda2, npoints)
    x = z[:, 0]
    pathlengths = z[:, 1]
    pathlengths = np.interp(xi, x, pathlengths)
    return pathlengths


def ioi_get_extinctions(lambda1, lambda2, npoints):
    """
    Returns the extinction coefficients (epsilon) for Hbo and HbR as a function of wavelength between lambda1 and lambda2
    Values in 1/(cm*M) by Scott Prahl at http://omlc.ogi.edu/spectra/hemoglobin/index.html are stored in the Prahl_values.txt file.
    Parameters
    ----------
    lambda1/lambda2: scalar
        Wavelengths between which to return extinction values
    npoints: int
        Number of sampling points between lambda1/2.
    Returns
    -------
    ext_HbO/HbR: 1darrays
        Extinction values for HbO and HbR in 1/(cm*M) between lambda1 and lambda2
    """
    with open(r"C:\Users\gabri\Documents\Université\Maitrise\Projet\Widefield-Imaging-Acquisition\analysisPipeline\Prahl_values.txt", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        prahl_values = []
        for row in reader:
            prahl_values.append(list(map(float, row)))
    E = np.array(prahl_values)
    E[:,1:3] = E[:,1:3]*2.303 #correction for neperian form of the B-L law
    xi = np.linspace(lambda1, lambda2, npoints)
    x = E[:,0]
    y_HbO = E[:,1]
    y_HbR = E[:,2]
    ext_HbO = np.interp(xi, x, y_HbO)
    ext_HbR = np.interp(xi, x, y_HbR)
    return ext_HbO, ext_HbR


def ioi_epsilon_pathlength(lambda1, lambda2, npoints, baseline_hbt, baseline_hbo, baseline_hbr, filter):
    """
    Returns the extinction coefficient*pathlength curve for Hbo and HbR as a function of wavelength between lambda1 and lambda2
    Parameters
    ---------
    lambda1/2: scalars
        Wavelengths between which the system specs are defined.
    npoints: int
        Number of wavelength sampling points in system specs
    baseline_hbt/o/r: scalars
        Baseline concentrations of HbT, HbO and HbR in the brain, in uM
    filter: boolean
        Specify if the fluoresence emission filter was in place.
    Returns
    -------
    eps_pathlength: 2darray
        2d matrix of the epsilon*pathlength values for both imaging wavelengths (rows) and chromophores (columns) in 1/M.
        This matrix is used to solve the modified Beer-Lambert equation for HbO and HbR concentration changes.
    """
    os.chdir(r"C:\Users\gabri\Documents\Université\Maitrise\Projet\Widefield-Imaging-Acquisition")
    wl = np.linspace(lambda1, lambda2, npoints)
    # c_camera
    QE_moment = np.loadtxt(r"analysisPipeline\specs sys optique\QE_moment_5px.csv", delimiter=';')
    p = np.poly1d(np.polyfit(QE_moment[:,0], QE_moment[:,1], 10))
    c_camera = p(wl)/np.max(p(wl))
    QE_moment, p = None, None
    # c_led
    FBH530 = np.loadtxt(r"analysisPipeline\specs sys optique\FBH530-10.csv", skiprows=1, usecols=(0, 2), delimiter=';')
    f = interp1d(FBH530[:,0], FBH530[:,1])
    c_FBH530 = f(wl)/np.max(f(wl))
    FBH630 = np.loadtxt(r"analysisPipeline\specs sys optique\FBH630-10.csv", skiprows=1, usecols=(0, 2), delimiter=';')
    f = interp1d(FBH630[:,0], FBH630[:,1])
    c_FBH630 = f(wl)/np.max(f(wl))
    c_led = np.array([c_FBH530, c_FBH630])
    FBH530, FBH630, c_FBH530, c_FBH630, f = None, None, None, None, None 
    c_tot = baseline_hbt*10**-6  # Rough baseline concentrations in M
    c_pathlength = ioi_path_length_factor(lambda1, lambda2, npoints)
    c_ext_hbo, c_ext_hbr = ioi_get_extinctions(lambda1, lambda2, npoints)
    # Create vectors of values for the fits
    CHbO = baseline_hbo/baseline_hbt*c_tot*np.linspace(0, 1.5, 16) #in M
    CHbR = baseline_hbr/baseline_hbt*c_tot*np.linspace(0, 1.5, 16)
    # In this computation we neglect the fact that pathlength changes with total concentration
    # (it is fixed for a Ctot of 100e-6)
    eps_pathlength = np.zeros((2, 2))
    IHbO = np.zeros(np.shape(CHbO))
    IHbR = np.zeros(np.shape(CHbR))
    for iled in range(2):
        for iconc in range(len(CHbO)):
            IHbO[iconc] = np.sum(c_camera*c_led[iled]*np.exp(-c_ext_hbo*c_pathlength*CHbO[iconc]))
            IHbR[iconc] = np.sum(c_camera*c_led[iled]*np.exp(-c_ext_hbr*c_pathlength*CHbR[iconc]))
        IHbO = IHbO/np.max(IHbO)
        IHbR = IHbR/np.max(IHbR)
        # Compute effective eps
        # plt.plot(c_camera*c_led[iled]*np.exp(-c_ext_hbr*c_pathlength*CHbO[iconc]), 'r.')
        # plt.plot(c_camera*c_led[iled]*np.exp(-c_ext_hbo*c_pathlength*CHbR[iconc]), 'g.')
        p1 = np.polyfit(CHbO, -np.log(IHbO), 1)
        p2 = np.polyfit(CHbR, -np.log(IHbR), 1)
        HbOL = p1[0]
        HbRL = p2[0]
        eps_pathlength[iled, 0] = HbOL
        eps_pathlength[iled, 1] = HbRL
    # print(eps_pathlength)
    return eps_pathlength

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
    """_summary_

    Args:
        frames (array): 3D array of one type of data, ex HbO, HbR, or HbT
        Hb (str): type of data
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



def lowpass_filter(sig:list, cutoff:float=1, fs:float=10, order:int=5)->list:
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


def regress_drift(sig:list, time:list, save_path, wl:int=530)-> list:
    """Prepares raw data to calculate HbO and HbR: removes 
        drift if any, and normalizes around 1

    Args:
        sig (list): 1D array containing signal
        time (list): 1D array containing time
        wl (int): wavelength of light corresponding to data, necessary for saving data as npy. Defaults to 530
        filter (bool): activate Defaults to False
        
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


def prepToCompute(frames:list, wl:int=530, correct_motion=False, bin_size=None, regress=False):
    """_summary_

    Args:
        frames (list): _description_
        wl (int, optional): _description_. Defaults to 530.
        correct_motion (bool, optional): _description_. Defaults to False.
        filter (bool, optional): _description_. Defaults to False.
        bin_size (_type_, optional): _description_. Defaults to None.
        regress (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
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

def convertToHb(data_green, data_red):
    """_summary_

    Args:
        data_green (_type_): _description_
        data_red (_type_): _description_

    Returns:
        _type_: _description_
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


def dHb_pipeline(data_path, save_path, correct_motion=True, bin_size=3, filter_sigma=2.5, regress=True):
    """_summary_

    Args:
        data_path (_type_): _description_
        save_path (_type_): _description_
        correct_motion (bool, optional): _description_. Defaults to True.
        bin_size (int, optional): _description_. Defaults to 3.
        filter (bool, optional): _description_. Defaults to True.
        cutoff (float, optional): _description_. Defaults to 0.2.
        regress (bool, optional): _description_. Defaults to True.
    """
    # process green
    print("Loading green data")
    green = create_npy_stack(data_path + "\\530", data_path, 530, saving=False)
    # green = np.load(data_path + "\\530_rawStack.npy")
    print("Green data loaded")
    green = prepToCompute(green, wl=530, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
    green = np.save(data_path + "\\530preped.npy", green)
    green = None
    print("Green data preped and saved")

    # process red
    print("Loading red data")
    red = create_npy_stack(data_path + "\\625", data_path, 625, saving=False)
    # red = np.load(data_path + "\\625_rawStack.npy")
    print("Red data loaded")
    red = prepToCompute(red, wl=625, correct_motion=correct_motion, bin_size=bin_size, regress=regress)
    red = np.save(data_path + "\\625preped.npy", red)
    red = None
    print("Red data preped and saved")

    # convert to hb
    print("Converting to dHb")
    green = np.load(data_path + "\\530preped.npy")
    red = np.load(data_path + "\\625preped.npy")
    d_HbO, d_HbR = convertToHb(green, red)
    d_HbT = d_HbO+d_HbR
    # resample pixel values
    d_HbO = resample_pixel_value(d_HbO, 16).astype(np.uint16)
    d_HbR = resample_pixel_value(d_HbR, 16).astype(np.uint16)
    d_HbT = resample_pixel_value(d_HbT, 16).astype(np.uint16)
    Hb = np.array((d_HbO, d_HbR, d_HbT))
    # filter if needed
    if filter is not None:
        print("Filtering")
        Hb = gaussian_filter(Hb, sigma=filter_sigma, axes=(1))
    # save processed data as npy
    np.save(save_path + "\\computedHb.npy", Hb)
    # save as tiff
    print("Saving processed Hb")
    data_types = ['HbO', 'HbR', 'HbT']
    for frames, typpe in zip(Hb, data_types):
        try:
            os.mkdir(save_path + "\\"  + typpe)
        except FileExistsError:
            print("Folder already created")
        save_as_tiff(frames, typpe, save_path + "\\" + typpe)

    print("Done")

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    data_path = filedialog.askdirectory()
    save_path = data_path
    dHb_pipeline(data_path, save_path, bin_size=None)
