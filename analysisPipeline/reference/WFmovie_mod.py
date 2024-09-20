import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import RectangleSelector, PolygonSelector, EllipseSelector
import json
import csv
import os
from PIL import Image
# from EasyROI import EasyROI
from plantcv import plantcv as pcv
from skimage.registration import phase_cross_correlation
import scipy.ndimage as ndi
from scipy.ndimage import shift, gaussian_filter, median_filter
from scipy.io import loadmat
from scipy.stats import zscore
from scipy.signal import cheby1, filtfilt
from scipy.interpolate import interp1d
import cv2
from tqdm import tqdm
import sys
from threading import Thread
import math
from pathlib import Path

class WFmovie():
    def __init__(self, folder_path=None, channel=None, movie=None, memmap = False):
        """
        Constructor for the ImagingMovie class.
        Arguments:
        ----------    
        folder_path: string 
            Path to folder containing the data.
        channel: string 
            Channel to load, either 'green', 'red', 'ir' or 'blue'.
        movie: ndarray (optional)
            3d array containing movie to analyse. First dimension must be time and dimensions 2 and 3 the movie frames. If no array is 
            specified, folder_path for the experiment and channel for the movie to analyse must be provided. If movie data are specified, 
            folder_path can still be provided to extract stimulation data from a given experiment in the self.stim attribute.
        memmap: boolean (optional) 
            Specify if data should be accessed via a memory map file, or directly loaded. The default is False.
        """
        if folder_path is not None:
            self.folder_path = Path(folder_path)
            self.channel = channel
            self.movie = movie 
            with open(self.folder_path / "metadata.json", 'r') as metadata_file:
                self.metadata = json.load(metadata_file)
            self.freq = self.metadata["Framerate"]
            stim_signal = np.load(self.folder_path / "stim_signal.npy")
            stim_blocks = create_stim_blocks(stim_signal)
            if movie is None:
                self.data_file = self.folder_path / "".join([channel, "Chan.npy"])
                if memmap is True:
                    self.data = np.lib.format.open_memmap(self.data_file, mode='r+')
                else:
                    self.data = np.load(self.data_file)
            else:
                self.data = movie                            
        else:
            self.data = movie
        self.nframes = np.shape(self.data)[0]
        self.nrows = np.shape(self.data)[1]
        self.ncols = np.shape(self.data)[2]
        self.ref_frame = self.data[0]
        if folder_path is not None:
            self.stim = resample(np.transpose(stim_blocks), 3000, self.freq, self.nframes)

    def correct_motion(self, select_roi=True):
        """Applies dft registrarion algorithm to correct for motion.
        Args:
            select_roi (bool, optional): Whether dft registration is only applied to a selected ROI.
                                         Defaults to True.
        """
        if select_roi:
            data = self.data
            # img_raw = np.uint8(self.data[0] / np.max(self.data[0]) * 255)
            # roi_helper = EasyROI(verbose=True)
            # if img_raw.shape[0] < 300:
            #     img = Image.fromarray(img_raw)
            #     img = img.resize(
            #         (2*img_raw.shape[0], 2*img_raw.shape[1]), Image.NEAREST)
            #     img_raw = np.array(img)
            # rectangle_roi = roi_helper.draw_rectangle(img_raw, 1)
            # if img_raw.shape[0] < 300:
            #     rectangle_roi['roi'][0].update(
            #         {i: int(j/2) for i, j in rectangle_roi['roi'][0].items()})
            # data = roi_helper.crop_roi(self.data, rectangle_roi)[0]
        else:
            data = self.data
        img1 = data[0]
        for i, img in enumerate(data[1:]):
            img2 = img
            shifted, error, diffphase = phase_cross_correlation(
                img1, img2, upsample_factor=10)
            corrected_image = shift(self.data[i], shift=(
                shifted[0], shifted[1]), mode='reflect')
            self.data[i] = corrected_image

    def normalize_by_mean(self):
        """Normalize each frame by the movie mean.
        """
        self.data = self.data / np.mean(self.data, axis=0)

    def zscore(self):
        """Convert each pixel to its temporal z-score.
        """
        self.data = zscore(self.data, axis=0, nan_policy='omit')

    def gaussian_filt(self, sigma):
        """Apply spatial gaussian filter to each frame.

        Args:
            sigma (float): Standard deviation of the gaussian filter.
        """
        for i, frame in enumerate(self.data[:]):
            self.data[i] = gaussian_filter(frame, sigma)

    def bandpass_filt(self, cutOnFreq, cutOffFreq, mask=None):
        """Apply temporal bandpass filter to each pixel.

        Args:
            cutOnFreq (float): Cut-off frequency for the low-pass filter.
            cutOffFreq (float): Cut-off frequency for the high-pass filter.
            mask (ndarray): Mask associated with the data.
        """
        if mask is None:
            mask = np.ones((self.nrows,self.ncols))
        if cutOffFreq > self.freq / 2:
            cutOffFreq = self.freq / 2
            print(
                'Cut off frequency did not respect Nyquist criteria and was reduced to', cutOffFreq, 'Hz.')
        initial_matrix = convert_to_2d_matrix(self.data,mask)
        b, a = cheby1(1, 3, np.array(
            [cutOnFreq, cutOffFreq]) / (self.freq), btype='bandpass')
        filtered_matrix = filtfilt(b, a, initial_matrix, axis=0)
        self.convert_matrix_to_movie(filtered_matrix, mask)
            
    def med_temp_filt(self,width):
        """
        Apply a temporal median filter to each pixel.

        Args:
            width (odd int): Temporal width of the median filter (in number of samples)
        """
        if (width % 2)==0:
            print(r'Filter width must be odd')
            return
        self.data = median_filter(self.data,(width,1,1), mode='nearest')

    def create_mask(self,frame = None):
        """Create a binary mask.
        Args:
            frame (ndarray): Frame from which to create the mask (default: first data frame)
        Returns:
            mask (uint8 ndarray): Binary mask 
        """
        if frame is None:
            frame = self.ref_frame
        def create_mask_from_verts(*args,im=frame):
            img_uint8 = np.uint8(im*255/np.max(im))
            roi_contour, roi_hierarchy = pcv.roi.custom(img=img_uint8, 
                                                   vertices=args[0])
            mask_private = pcv.roi.roi2mask(img=img_uint8, contour=roi_contour)
            mask_private = mask_private/np.max(mask_private)
            mask.append(mask_private)
        fig, ax = plt.subplots()
        ax.imshow(frame)
        plt.axis('off')
        plt.xlim(0,np.shape(frame)[1])
        plt.ylim(0,np.shape(frame)[0])
        selector = PolygonSelector(ax, create_mask_from_verts)
        ax.invert_yaxis() #PolygonSelector inverts the y axis relative to the imshow function
        plt.show()
        mask = []
        print("Click on the figure to create a polygon.")
        print("Hold the 'shift' key to move all of the vertices.")
        print("Hold the 'ctrl' key to move a single vertex.")
        def waiting():
            wU = True
            while wU == True:
                if len(mask) != 0:
                    wU = False
                    plt.close(fig)
                plt.pause(4)
                
        waiting_thread = Thread(target=waiting())
        waiting_thread.start() 
        return mask[0]

    def apply_mask(self, mask):
        """Applies a mask to the data.

        Args:
            mask (ndarray): Mask to apply to the data.
        """
        self.data = self.data * mask

    def convert_matrix_to_movie(self,matrix,mask):
        """Convert a movie presented under 2D format to the original 3D format. 
           This method updates the object's data attribute.
        Args:
            matrix (time by pixels ndarray): Frames presented under time by pixel format
            mask (binary ndarray): region of the pixels under matrix format        
        """
        mask1D = mask.reshape(self.nrows*self.ncols)
        masked_indexes = np.nonzero(mask1D)[0]
        frame = np.zeros(self.nrows*self.ncols)
        nframes = np.shape(matrix)[0]
        for i in range(nframes):
            frame[masked_indexes] = matrix[i]
            self.data[i] = frame.reshape((self.nrows,self.ncols))
            
    def rotate_data(self, number_of_rots):
        """Rotate data by 90, 180 or 270 degrees

        Args:
            number_of_rots (int): Number of consecutive 90 degrees rotation to do.
        """
        self.data = np.rot90(self.data, k=number_of_rots, axes=[1, 2])

    def show_movie(self, freq=None, filename=None):
        """Show the movie.

        Args:
            freq (int, optional): Framerate of the movie. If no value is specified, the acquisition frequency 
                                  is used.
            filename (str, optional): Name of the mp4 file where the movie is saved. If none is specified, 
                                      the movie is not saved.
        Returns:
            ani: animation object. Display by calling plt.show()
        """
        if freq is None:
            freq = self.freq
        fig, ax = plt.subplots()
        plt.axis('off')
        vmin = np.min(self.data[0])
        vmax = np.max(self.data[0])
        def animate(i):
            ax.clear()
            ax.imshow(self.data[i,:,:],vmin=vmin,vmax=vmax)
        ani = animation.FuncAnimation(fig, animate,frames=self.nframes,interval=1/freq)
        if filename is not None:             
           save_array_as_avi(self.folder_path / filename, self.data)
        return ani
    
    def get_timeseries(self,ROI=None,frames=None):
        """
        Extract time series from ROI.
        Args:
            ROI (2D binary array, default None): Mask that specifies the ROI. If None, an interactive 
            window appears for manual ROI selection.
            frames (3d array,default None): Specify the frames from which to extract timeseries. If None, the object's data attribute is used.            
        Returns:
            mean_timeseries (vector)
        """
        if ROI is None:
            ROI = self.create_mask()
        timeseries = convert_to_2d_matrix(self.data,ROI)
        mean_timeseries = np.mean(timeseries,axis=1)
        return mean_timeseries
        
    def detrend(self,mask=None):
        """ Remove linear trends from data.
        Args:
            mask (2D binary array, optional): Limit detrending to specified region. 
        """
        if mask is None:
            mask = np.ones((self.nrows,self.ncols))
        data = convert_to_2d_matrix(self.data,mask)
        frame_index = np.arange(0,self.nframes,1)
        frame_index = frame_index[:,None] #Make Nx1 vector
        regressed_data = regress_timeseries(data,frame_index,offset=True)
        self.convert_matrix_to_movie(regressed_data, mask)
        
    def gsr(self,ROI = None, mask=None):
        """ Perform global signal regression.
        Args:
            ROI: (2D binary array, optional): Pre-selected region from which to extract global signal
            mask (2D binary array, optional): Limit gsr to specified region. 
        """
        if ROI is None:
            ROI = self.create_mask(self.ref_frame)
        if mask is None:
            mask = np.ones(self.nrows,self.ncols)
        data = convert_to_2d_matrix(self.data,mask)
        global_signal = self.get_timeseries(ROI)
        regressed_data = regress_timeseries(data,global_signal)
        self.convert_matrix_to_movie(regressed_data, mask)
        
    def speckle_contrast(self, kernel_size=7,temporal=False,spatiotemp=False):
        """ Convert movie frames to speckle contrast maps
        Args:
            kernel_size (int, default = 7): Width of the kernel used to compute the speckle contrast statistics
            temporal (boolean, default = False): Set to True to obtain the temporal speckle contrast.
        """
        if temporal is False and spatiotemp is False:
            kernel = np.ones((kernel_size,kernel_size))/kernel_size**2
            for i,frame in enumerate(self.data):
                mean = ndi.correlate(frame,kernel,mode = 'nearest')
                std = np.sqrt(ndi.correlate(frame*frame,kernel,mode = 'nearest')-mean**2)
                K = std/mean
                self.data[i] = K
        elif temporal is True:
            data_matrix = convert_to_2d_matrix(self.data)
            kernel = np.ones((kernel_size,1))/kernel_size
            mean = ndi.correlate(data_matrix,kernel,mode='nearest')
            std = np.sqrt(ndi.correlate(data_matrix*data_matrix,kernel,mode = 'nearest')-mean**2)
            K = std/mean
            self.convert_matrix_to_movie(K,np.ones(np.shape(self.data[0])))
        elif spatiotemp is True:
            try:
                kernel = np.ones((kernel_size[1],kernel_size[0],kernel_size[0]))/(kernel_size[0]**2*kernel_size[1])
            except (AttributeError,TypeError):
                raise AssertionError('Kernel size should be a 2-value array or tuple')
                         
    def bin_frames(self):
        """Perform 2x2 binning of frames"""
        new_size = (round(self.nrows/2),round(self.ncols/2))
        binned_frames = np.zeros((self.nframes,new_size[0],new_size[1]))
        for i,frame in enumerate(self.data):
            image = Image.fromarray(frame)
            image = image.convert('F') #F = 32-bit floating point
            binned_image = image.resize((new_size[1],new_size[0]), resample=Image.Resampling.BILINEAR)
            binned_frames[i] = np.array(binned_image)
        self.data = binned_frames
        self.nrows = new_size[0]
        self.ncols = new_size[1]
        
    def average_response(self,stim_channel, preStim, postStim, stim_index = None, timeseries = False, ROI = None):
        """
        Extract stimulus block-averaged responses
        Parameters:
        -----------
        stim_channel: int (1 or 2)
            Indicate which stimuation channel to consider.
        preStim: scalar 
            Lenght of pre-stimulation period (s). The average signal frame during this period is taken as the baseline for the stimulation period.
        postStim: scalar
            Length of post-stimulation period (s).
        stim_index: 1darray (optional, default None) 
            Specify the index of particular stimuli to consider. When None, all stimuli are considered.
        timeseries: boolean (optional, default False)
            Return a time series vector of the average signal within ROI instead of a 3d array.
        ROI: 2darray, default None: 
            Specify the ROI from which to extract the time series. If timeseries = True and ROI = None, an interactive ROI selection window appears.
        Returns:
        --------
        avg_response: ndarray 
            Frames of the block-averaged response. 1d time series vector if timeseries=True.   
        """
        stim = self.stim[:,stim_channel-1]
        offsets = np.argwhere(np.diff(stim)<0) #indexes of last frames during stim
        onsets = np.argwhere(np.diff(stim)>0)+1 #indexes of first frames during stim
        if stim_index is not None:
            offsets = offsets[stim_index-1]
            onsets = onsets[stim_index-1]   
        npost_frames = math.floor(postStim*self.freq)
        npre_frames = math.floor(preStim*self.freq)
        offsets = (offsets+npost_frames).astype(int)
        onsets = (onsets-npre_frames).astype(int)
        nframes = np.min(offsets-onsets)+1 #Min value is taken since there may be minor differences in frame number between blocks
        nstim = np.size(offsets)
        avg_response = np.zeros((nframes,self.nrows,self.ncols))
        for i in np.arange(nstim):
            baseline = np.mean(self.data[onsets[i,0]:onsets[i,0]+npre_frames],axis=0)
            response = self.data[onsets[i,0]:onsets[i,0]+nframes]/baseline
            avg_response = avg_response+response
        avg_response = avg_response/nstim
        if timeseries:
            timeseries_matrix = convert_to_2d_matrix(avg_response,ROI)
            avg_response = np.mean(timeseries_matrix,axis=1)
        return avg_response
        
            
def create_channel(folder_path, channel, metadata_file="metadata.json", binning=False, region=None):
    """
    Creates a data file and a metadata file for a specific channel.
    Arguments:
    ----------
    folder_path: string 
        Path to the folder containing the .npy data files.
    channel: string 
        Name of the channel to create between 'ir', 'red' or 'green' or 'blue'.
    metadata_file: string 
        Name of the metadata file. Default is "metadata.json".
    binning: boolean (optional) 
        Performs 2x2 binning of data if True. Defaults is False.
    region: tuple (optional) 
        Specify image region as (xmin,xmax,ymin,ymax). Default takes whole image.
    """
    folder_path = Path(folder_path)
    with open(folder_path / metadata_file, 'r') as metadata_file:
        metadata = json.load(metadata_file)
    size = [int(metadata['Dimensions'][1]), int(metadata['Dimensions'][0])]
    if region is not None:
        x1 = region[0]
        x2 = region[1]
        y1 = region[2]
        y2 = region[3]
    else:
        x1 = 0
        x2 = size[1]
        y1 = 0
        y2 = size[0]
    size = [y2-y1, x2-x1]
    original_size = size
    lights = metadata['Lights']
    nlights = len(lights)
    freq = metadata['Framerate'] / nlights
    i_chan = lights.index(channel)
    number_datafiles = len(os.listdir(folder_path / 'data'))
    datafiles = [f'{i}.npy' for i in range(number_datafiles)]
    last_file = np.load(folder_path / 'data' / datafiles[-1])
    last_file_length = len(last_file)
    nframes_per_file = int(1200/nlights) #each initial .npy data file contains 1200 frames
    del last_file
    length = nframes_per_file * (len(datafiles) - 1) + \
        int(last_file_length / nlights)
    # Create channel file
    if binning:
        size = (original_size[0] // 2, original_size[1] // 2)
    channel_data = np.lib.format.open_memmap(folder_path / "".join([channel, "Chan.npy"]), shape=(length, size[0], size[1]),
                                             mode='w+', dtype='single')
    # Write data to channel file
    for i, filename in enumerate(datafiles[:-1]):
        file = np.lib.format.open_memmap(folder_path / 'data' / filename, 'r')[i_chan::nlights,y1:y2,x1:x2]
        if binning:
            binned_file = []
            for frame in file[:]:
                image = Image.fromarray(frame)
                image = image.convert('I')
                binned_image = image.resize(
                    (size[1], size[0]), resample=Image.BILINEAR)
                binned_file.append(np.array(binned_image.convert('I;16')))
            channel_data[i*nframes_per_file:(i+1)*nframes_per_file] = binned_file
        else:
            channel_data[i*nframes_per_file:(i+1)*nframes_per_file] = file
    last_file = np.lib.format.open_memmap(folder_path / 'data' / datafiles[-1], 'r')[i_chan::nlights,y1:y2,x1:x2]
    nframes_last_file = np.shape(last_file)[0]
    if binning:
        last_binned_file = []
        for frame in last_file:
            image = Image.fromarray(frame)
            image = image.convert('I')
            binned_image = image.resize(
                (size[1], size[0]), resample=Image.BILINEAR)
            last_binned_file.append(np.array(binned_image.convert('I;16')))
        channel_data[-nframes_last_file:] = last_binned_file
    else:
        channel_data[-nframes_last_file:] = last_file
    # Create metadata file
    channel_variables = {
        "datLength": length,
        "datSize": size,
        "Freq": freq,
    }
    with open(folder_path / "".join(['metadata_', channel, '.json']), 'wt') as channel_metadata:
        channel_metadata.write(json.dumps(channel_variables))


def create_stim_blocks(stim):
    """
    Creates a stimulus blocked signal from the raw stimulation voltage data.
    Parameters
    ----------
    stim : 2darray
        stimulus voltage time series (channels by time)
    Returns
    -------
    stim_blocks : 2darray
        stimulus time series blocked by events
    """
    THRES = 0.9  # threshold in pct of max value for stim to be considered ON
    SAMPLING_FREQ = 3000
    dt = 1/SAMPLING_FREQ
    MAX_INT = 2  # max time interval [s] between two ON values to be considered on the same block
    n_max = int(MAX_INT/dt)
    stim_blocks = np.zeros(np.shape(stim))
    for channel in range(np.shape(stim)[0]):
        stim_ = stim[channel,:]
        stim_blocks_ = np.zeros(np.shape(stim_))
        on_indices = np.argwhere(stim_ > THRES*np.max(stim_))
        if on_indices.size != 0:
            diff_on_indices = np.diff(on_indices, axis=0)
            stim_blocks_[int(on_indices[0]):int(on_indices[-1])] = 1
            for i in np.argwhere(diff_on_indices > n_max)[:, 0]:
                stim_blocks_[int(on_indices[i])+1:int(on_indices[i+1])] = 0
            stim_blocks[channel,:] = stim_blocks_
    return stim_blocks


def resample(y1, f1, f2, n):
    """
    Resamples binary signal y1, measured at a sampling rate f1, to the frequency f2 for n points by linear interpolation. If n is large
    enough to extend the resampled signal beyond the duration of y1, the extra values take on the last value of y1.
    Parameters
    ----------
    y1 : ndarray
        time series vector. If multiple time series, combine as a (time x signal) 2d array 
    f1/2 : scalar
        sampling rates of y1/resampled rate
    n : int
        number of desired points for the resampled signal
    Returns
    -------
    Y : ndarray
     resampled signal(s) y1, of length n and sampling rate f2  

    """
    dt2 = 1/f2
    dt1 = 1/f1
    y1_length = (np.shape(y1)[0]-1)*dt1
    t2 = np.arange(0, n*dt2, dt2)
    t1 = np.arange(0, y1_length+dt1, dt1)
    if np.ndim(y1) == 1:
        Y = np.zeros(n)
    else:
        Y = np.zeros((n,np.shape(y1)[1]))
    for i in np.arange(n):
        sampled_time = t2[i]
        closest_time_index = np.argmin(np.abs(t1-sampled_time))
        t1_1 = t1[closest_time_index]
        y1_1 = y1[closest_time_index]
        if t1_1 != t1[-1]:
            if sampled_time >= t1_1:
                t1_2 = t1[closest_time_index+1]
                y1_2 = y1[closest_time_index+1]
                Y[i] = y1_1+(y1_2-y1_1)/(t1_2-t1_1)*(sampled_time-t1_1)
            else:
                t1_2 = t1[closest_time_index-1]
                y1_2 = y1[closest_time_index-1]
                Y[i] = y1_2+(y1_1-y1_2)/(t1_1-t1_2)*(sampled_time-t1_2)
        else:
            Y[i] = y1[-1]
    return Y


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
    QE_moment = np.loadtxt("analysisPipeline\specs sys optique\QE_moment_5px.csv", delimiter=';')
    p = np.poly1d(np.polyfit(QE_moment[:,0], QE_moment[:,1], 10))
    c_camera = p(wl)/np.max(p(wl))
    QE_moment, p = None, None
    # c_led
    FBH530 = np.loadtxt("analysisPipeline\specs sys optique\FBH530-10.csv", skiprows=1, usecols=(0, 2), delimiter=';')
    f = interp1d(FBH530[:,0], FBH530[:,1])
    c_FBH530 = f(wl)/np.max(f(wl))
    FBH630 = np.loadtxt("analysisPipeline\specs sys optique\FBH630-10.csv", skiprows=1, usecols=(0, 2), delimiter=';')
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


def convert_to_hb(R_green, R_red, filter=False):
    """
    Convert red and green relative reflectance data to changes in HbO and HbR concentrations
    Parameters
    ----------
    R_green/R_red : ndarray
        Green/red relative reflectance data. Values of 1 correspond to no change relative to a pre-defined baseline.
    filter : boolean
        Specify if the fluoresence emission filter was in place during acquisition. The default is False.
    Returns
    -------
    d_HbO/d_HbR : ndarrays
        HbO and HbR concentration changes in M. The output shape is the same as green/red.
    """
    #Parameters
    lambda1 = 450 #nm
    lambda2 = 700 #nm
    npoints = 1000
    baseline_hbt = 100 #uM
    baseline_hbo = 60 #uM
    baseline_hbr = 40 #uM
    
    eps_pathlength = ioi_epsilon_pathlength(lambda1, lambda2, npoints, baseline_hbt, baseline_hbo, baseline_hbr, filter)
    Ainv = np.linalg.pinv(eps_pathlength)
    ln_green = -np.log(R_green.flatten())
    ln_red = -np.log(R_red.flatten())
    ln_R = np.concatenate((ln_green.reshape(1,len(ln_green)),ln_red.reshape(1,len(ln_green))))
    Hbs = np.matmul(Ainv, ln_R)
    d_HbO = Hbs[0].reshape(np.shape(R_green))
    d_HbR = Hbs[1].reshape(np.shape(R_green))
    # Protection against aberrant data points
    np.nan_to_num(d_HbO, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(d_HbR, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    
    return d_HbO, d_HbR


def save_array_as_avi(path, array, codec='XVID', fps=30, color=False):
     """3D array shape: (t, y, x) or (z, y, x)"""
     if array.dtype != 'uint8':
         array = array.astype('double')/4096*255
         array = array.astype(np.uint8)
     writer = cv2.VideoWriter_fourcc(*codec)
     video = cv2.VideoWriter(path, writer, float(fps), (array.shape[2], array.shape[1]), isColor=color)
     N_frames = int(array.shape[0])
     for i in tqdm(range(N_frames), file=sys.stdout):
         video.write(array[i])
     video.release()

def regress_timeseries(y,x,intercept = True,offset = False):
    """ Regress out the linear contribution of a regressor on time series data
    Args:
        y (ndarray): Time series specified as Nx1 vector. For M time series, specify as NxM array. 
        x (1D vector): NX1 time series to regress out of data.
        intercept (default = True) Specify wether or not to include an intercept term in the regressor.
        offset (default = False) Option to re-offset the regressed data at the estimated intercept
    Returns:
        eps (ndarray): Regressed data
    """
    if np.ndim(y)==1: #Make sure time series are presented as NX1 vectors
        y = y[:,None]
    N = np.shape(y)[0]
    if np.shape(x)[0] != N:
        raise Exception('Regressor must be of height N, with N the number of time points.')
    if np.ndim(x)==1: 
        x = x[:,None]
    if offset and intercept is False:
        intercept = True
    if intercept:
        x = np.insert(x,0,np.ones((1,N)),axis = 1)
    x_inv = np.linalg.pinv(x)
    beta = np.matmul(x_inv,y)
    y_est = np.matmul(x,beta)
    eps = y-y_est
    if offset:
        eps = eps+np.matmul(x[:,0][:,None],beta[0,:][None,:])
    return eps

def convert_to_2d_matrix(X, mask=None):
    """
    Convert 3d movie data formatted as time by frames to a 2D  time by pixel matrix.
    Parameters:
    -----------
    X: 3darray
        Data to convert.
    mask: binary ndarray (optional)
        Consider only pixels within mask.
    Returns:
    --------
    Y: 2darray 
        2D matrix of the data with each column representing the time-course of a different pixel.
    """
    nframes = np.shape(X)[0]
    nrows = np.shape(X)[1]
    ncols = np.shape(X)[2]
    Y = np.reshape(X,(nframes,nrows*ncols))
    if mask is not None:
        mask1D = np.reshape(mask,(nrows*ncols))
        masked_indexes = np.nonzero(mask1D)[0]
        Y = Y[:, masked_indexes]
    return Y

def _figure_to_array(fig):
    """Private function used in save_figures_as_gif"""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    array.shape = (h, w, 4)
    array = np.roll(array, 3, axis=2)
    return array

def save_figures_as_gif(path, figures, fps=30):
    """
    Save series of fig objects as a .gif file
    Parameters
    ----------
    path : str
        Path of desired save location
    figures : list
        List of figures to save
    fps : scalar
        gif frame rate per second (default 30)
    """
    images = []
    for figure in figures:
        array = _figure_to_array(figure)
        image = Image.fromarray(array)
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:], optimize=False, duration=1000/fps, loop=0)
        
    

if __name__ == "__main__":
    # pathBase = r"C:\Users\gabri\Desktop\testAnalyse\2024_07_18"
    # green = np.loadtxt(pathBase + "\\csv\\530.csv", skiprows=1, delimiter=',')[:,1]
    # red = np.loadtxt(pathBase + "\\csv\\625.csv", skiprows=1, delimiter=',')[:,1]

    # green_t = np.load(pathBase + "\\530ts.npy")
    # red_t = np.load(pathBase + "\\625ts.npy")

    # d_HbO, d_HbR = convert_to_hb(green, red)

    # plt.plot(green_t, d_HbO/np.mean(d_HbO))
    # plt.plot(green_t, d_HbR/np.mean(d_HbR))
    # plt.show()

    pass