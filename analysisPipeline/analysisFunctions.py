import numpy as np
from matplotlib import pyplot as plt

def deltaFoverF(data: list) -> list:
    """ Generates Delta F/F from raw data

    Args:
        data (list): list or array of data

    Returns:
        list: Delta F/F over time
    """

    return (data - np.mean(data))/np.mean(data)


def oxygenation(greenData: list, redData: list) -> tuple:
    """Generates a tuple that contains the variation of HbO and HbR concentrations over time

    Args:
        greenData (list): 530 nm absorption coefficient evolution in time
        redData (list): 625 nm absorption coefficient evolution in time

    Returns:
        tuple: variation of HbR and HbO over time (delta c_HbR, delta c_HbO)
    """
    def absorptionCoefficientVariation(intensity:list, wavelength:int=530) -> list:
        """Calculates the absorption coefficient variation for a specific wavelength

        Args:
            intensity (list): light signal over time
            wavelength (int, optional): wavelength of light, either 530 or 625 nm. Defaults to 530.

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
            
        mu = (-1/X)* np.log(intensity/intensity[0])
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
    
