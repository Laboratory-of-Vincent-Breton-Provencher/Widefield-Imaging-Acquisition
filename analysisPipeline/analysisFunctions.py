import numpy as np
import os
from tkinter import filedialog
from tkinter import *
from tqdm import tqdm
# from matplotlib import pyplot as plt


def splitChannels(FLAG405:bool = 1, FLAG470:bool = 1, FLAG530:bool = 1, FLAG625:bool = 1, FLAG785:bool = 1, fin:int=100) ->tuple:
    # folderPath:str,
    """_summary_

    Args:
        folderPath (str): _description_
        FLAG405 (bool, optional): _description_. Defaults to 1.
        FLAG470 (bool, optional): _description_. Defaults to 1.
        FLAG530 (bool, optional): _description_. Defaults to 1.
        FLAG625 (bool, optional): _description_. Defaults to 1.
        FLAG785 (bool, optional): _description_. Defaults to 1.
        fin (int, optional): _description_. Defaults to 100.

    Returns:
        tuple: _description_
    """

    root = Tk()
    root.withdraw()
    folderPath = filedialog.askdirectory()

    FLAGS = {"FLAG405":FLAG405, "FLAG470":FLAG470, "FLAG530":FLAG530, "FLAG630":FLAG625, "FLAG785":FLAG785}
    for flagName, flag in zip(FLAGS.keys, FLAGS.values):
        if flag:
            os.mkdir()


    filesList = os.listdir(folderPath)
    filesPaths = []
    for idx, file in enumerate(tqdm(filesList)):
        filesPaths.append(folderPath + "\\" + file)

    filesPaths.sort(key=lambda x: os.path.getmtime(x))
    filesPaths = filesPaths[1:]

    fin = 100

    filesList405 = filesPaths[0:fin:5]
    filesList470 = filesPaths[1:fin:5]
    filesList530 = filesPaths[2:fin:5]
    filesList625 = filesPaths[3:fin:5]
    filesList785 = filesPaths[4:fin:5]

    return (filesList405, filesList470, filesList530, filesList625, filesList785)


def normalizeData(data:list, dim:int=3) -> list:
    """_summary_

    Args:
        data (list): _description_
        dim (int, optional): dimension of data. Defaults to 3.

    Returns:
        list: _description_
    """
    datamin = np.min(data, axis=dim-1)[..., np.newaxis]
    datamax = np.max(data - datamin, axis=2)[..., np.newaxis]
    datamax[np.where(datamax == 0)] = 1
    return ((data - datamin)/(datamax))

def hemodynamicCorrection(blueData:list, violetData:list) -> list:
    """Applies hemodynamic correction to fluorescence data (blue) from isosbestic data (violet)

    Args:
        blueData (list): _description_
        greenData (list): _description_

    Returns:
        list: _description_
    """
    return blueData-violetData

def deltaFoverF(data: list, dim:int=3) -> list:
    """ Generates Delta F/F from raw data

    Args:
        data (list): list or array of data
        dim (int, optional): dimension of data. Defaults to 3.

    Returns:
        list: Delta F/F over time
    """
    avg = np.mean(data, axis=dim-1)[..., np.newaxis]
    avg[np.where(avg == 0)] = 1
    return (data - avg)/avg


def oxygenation(greenData: list, redData: list) -> tuple:
    """Generates a tuple that contains the variation of HbO and HbR concentrations over time

    Args:
        greenData (list): 530 nm absorption coefficient evolution in time
        redData (list): 625 nm absorption coefficient evolution in time

    Returns:
        tuple: variation of HbR and HbO over time (delta c_HbR, delta c_HbO)
    """
    def absorptionCoefficientVariation(intensity:list, wavelength:int=530, dim:int=3) -> list:
        """Calculates the absorption coefficient variation for a specific wavelength

        Args:
            intensity (list): light signal over time
            wavelength (int, optional): wavelength of light, either 530 or 625 nm. Defaults to 530.
            dim (int, optional): dimension of data. Defaults to 3.

        Returns:
            list: variation of mu_a coefficient over time
        """
        if wavelength == 530:
            X = 0.371713            # mm
        elif wavelength == 625:
            X = 3.647821            # mm
        else:
            print('Wrong wavelength input: 530 or 630 only')
            return None
        
        if dim == 3:
            iniIntens = intensity[:,:,0][..., np.newaxis]
  
        elif dim == 2:
            iniIntens = intensity[:,0][..., np.newaxis]

        elif dim == 1:
            iniIntens = intensity[0][..., np.newaxis]

        iniIntens[np.where(iniIntens == 0)] = 1
        # np.where(iniIntens == 0, 1, iniIntens)
        # print(np.shape(iniIntens))
        with open('iniIntens{}.txt'.format(wavelength), 'w') as outfile:
            for dataslice in iniIntens:
                np.savetxt(outfile, dataslice)
        mu = (-1/X)* np.log(intensity[:,:,1:]/iniIntens)
        return mu

    mu_530 = absorptionCoefficientVariation(greenData, 530)
    mu_625 = absorptionCoefficientVariation(redData, 625)

    eHbO_530 = 39956.8
    eHbR_530 = 39036.4
    eHbO_625 = 740.8
    eHbR_625 = 5763.4

    dc_HbR = (eHbO_530*mu_625 - eHbO_625*mu_530)/(eHbO_530*eHbR_625 + eHbO_625*eHbR_530)
    dc_HbO = (eHbR_530*mu_625 - eHbR_625*mu_530)/(eHbR_530*eHbO_625 + eHbR_625*eHbO_530)

    return (dc_HbR, dc_HbO)
    
splitChannels()