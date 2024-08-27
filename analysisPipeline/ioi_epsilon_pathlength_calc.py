import os
import numpy as np
from scipy.interpolate import interp1d
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


if __name__ == "__main__":
    pass