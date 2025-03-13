import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import butter, lfilter

sns.set_theme("poster", "ticks", font="Arial")


def test_files_shape(ttls_array, fluorescence_array):
    """Function that test the shape of the TTLs and the fluoresence data.

    Args:
        ttls_array (np.array): TTLs array
        fluorescence_array (np.array): Fluorescence array
    """
    if ttls_array.shape[0] != fluorescence_array.shape[0]:
        if np.abs(fluorescence_array.shape[0] - ttls_array.shape[0]) <= 2:
            idLast = min([fluorescence_array.shape[0],ttls_array.shape[0]])
            fluorescence_array = fluorescence_array[:idLast,:]
            ttls_array = ttls_array[:idLast]
            
        else:
            raise ValueError(f"""There is a huge mismastch between TTLs 
                             (n={ str(ttls_array.shape[0])}) and FP data 
                             (n={str(fluorescence_array.shape[0])}). 
                             Parsing of Ard might need to be fixed""")


def fluorescence_signal_length(excitation_fluorescence:np.array, isobestic_fluorescence:np.array):
    min_length_input_signals = np.min([excitation_fluorescence.size,
                                        isobestic_fluorescence.size])
    excitation_fluorescence = excitation_fluorescence[:min_length_input_signals]
    isobestic_fluorescence = isobestic_fluorescence[:min_length_input_signals]

    return excitation_fluorescence, isobestic_fluorescence

def process_fiber_photometry_data(excitation_fluorescence:np.array,
                                  isobestic_fluorescence:np.array,
                                  FPS:int=20) -> np.array:
    """Function that process the fiber photometry signal. We also define a
    Butterworth filter to remove slow changes in fluorescence, like bleaching.
    After we applied the filters, we substract the excitation and the isobestic
    signals. We end the processing with a small convolution filter.

    Args:
        excitation_fluorescence (np.array): Fluorescence from the excitation light
        isobestic_fluorescence (np.array): Fluorescence from the isobestic fluorescence
        FPS (int, optional): Acquisition frequency from the camera. Defaults to 20.

    Returns:
        np.array: Corrected fluorescence
    """

    b, a = butter(2, 0.01, "high", fs=FPS)

    filtered_excitation_fluorescence = lfilter(b, a, excitation_fluorescence - np.mean(excitation_fluorescence))
    filtered_isobestic_fluorescence = lfilter(b, a, isobestic_fluorescence - np.mean(isobestic_fluorescence))

    corrected_fluorescence = filtered_excitation_fluorescence - filtered_isobestic_fluorescence

    filtered_corrected_fluorescence = np.convolve(corrected_fluorescence,
                                                  np.ones(4)/4, mode='same')

    return filtered_corrected_fluorescence


if __name__ == "__main__":

    from scipy.stats import zscore
    
    paths = [
        r"Z:\aDaigle\Prediction AI\Data\Simultaneous recordings\RAW\01_18_02_2025\FP_2ch_2025-02-18T14_11_42.csv",
        r"Z:\aDaigle\Prediction AI\Data\Simultaneous recordings\RAW\02_18_02_2025\FP_2ch_2025-02-18T14_41_56.csv",
        r"Z:\aDaigle\Prediction AI\Data\Simultaneous recordings\RAW\03_18_02_2025\FP_2ch_2025-02-18T15_46_41.csv",
        r"Z:\aDaigle\Prediction AI\Data\Simultaneous recordings\RAW\01_19_02_2025\FP_2ch_2025-02-19T13_17_46.csv",
        r"Z:\aDaigle\Prediction AI\Data\Simultaneous recordings\RAW\01_20_02_2025\FP_2ch_2025-02-20T13_40_19.csv",
        r"Z:\aDaigle\Prediction AI\Data\Simultaneous recordings\RAW\02_19_02_2025\FP_2ch_2025-02-19T13_46_51.csv",
        r"Z:\aDaigle\Prediction AI\Data\Simultaneous recordings\RAW\02_20_02_2025\FP_2ch_2025-02-20T14_08_36.csv",
        r"Z:\aDaigle\Prediction AI\Data\Simultaneous recordings\RAW\03_19_02_2025\FP_2ch_2025-02-19T14_17_21.csv",
        r"Z:\aDaigle\Prediction AI\Data\Simultaneous recordings\RAW\03_20_02_2025\FP_2ch_2025-02-20T14_36_49.csv"]
    
    df = pd.read_csv(paths[4])

    TTLs = df[['V', 'B', 'TTL']]
    fiber_0_data = df['f0_ch1']
    fiber_1_data = df['f1_ch1']

    test_files_shape(TTLs, fiber_0_data)
    test_files_shape(TTLs, fiber_1_data)

    # Split the fluorescence
    isobestic_fluorescence_fiber_0 = fiber_0_data[0::2]
    excitation_fluorescence_fiber_0 = fiber_0_data[1::2]
    excitation_fluorescence_fiber_0, isobestic_fluorescence_fiber_0 = fluorescence_signal_length(excitation_fluorescence_fiber_0, isobestic_fluorescence_fiber_0)

    isobestic_fluorescence_fiber_1 = fiber_1_data[0::2]
    excitation_fluorescence_fiber_1 = fiber_1_data[1::2]
    excitation_fluorescence_fiber_1, isobestic_fluorescence_fiber_1 = fluorescence_signal_length(excitation_fluorescence_fiber_1, isobestic_fluorescence_fiber_1)


    corrected_fluorescence_signal_fiber_0 = process_fiber_photometry_data(excitation_fluorescence_fiber_0, isobestic_fluorescence_fiber_0)
    corrected_fluorescence_signal_fiber_1 = process_fiber_photometry_data(excitation_fluorescence_fiber_1, isobestic_fluorescence_fiber_1)

    FPS = 20    # Usual fiber photometry acquisition at 20 Hz
    time = np.arange(isobestic_fluorescence_fiber_0.size) / FPS

    # fig, ax = plt.subplots(2, 2, sharex=True, figsize=(10, 5))

    # ax[0, 0].set_title("GCaMP\nRaw signals")
    # ax[0, 0].plot(time, isobestic_fluorescence_fiber_0, label="Isobestic", c="purple")
    # ax[0, 0].plot(time, excitation_fluorescence_fiber_0, label="Excitation", c="b")
    # ax[0, 0].legend(fontsize=10)

    # ax[1, 0].set_title("Corrected fluorescence signal")
    # ax[1, 0].set_ylabel("z-score")
    # ax[1, 0].plot(time, zscore(corrected_fluorescence_signal_fiber_0), c="g")
    # ax[1, 0].set_xlabel("Time [s]")

    # ax[0, 1].set_title("GRAB-NE\nRaw signals")
    # ax[0, 1].plot(time, isobestic_fluorescence_fiber_1, label="Isobestic", c="purple")
    # ax[0, 1].plot(time, excitation_fluorescence_fiber_1, label="Excitation", c="b")
    # ax[0, 1].legend(fontsize=10)

    # ax[1, 1].set_title("Corrected fluorescence signal")
    # ax[1, 1].set_ylabel("z-score")
    # ax[1, 1].plot(time, zscore(corrected_fluorescence_signal_fiber_1), c="g")
    # ax[1, 1].set_xlabel("Time [s]")

    # plt.suptitle("Fiber photometry signals - 2 Fibers")
    # sns.despine()
    # plt.show()



    fig = plt.figure(figsize=(10, 5))

    plt.plot(time-865, (isobestic_fluorescence_fiber_1-np.mean(isobestic_fluorescence_fiber_1))*2+6, label="Isobestic", c="purple")
    plt.plot(time-865, (excitation_fluorescence_fiber_1-np.mean(excitation_fluorescence_fiber_1))*2+7, label="Excitation", c="b")
    plt.plot(time-865, zscore(corrected_fluorescence_signal_fiber_1), label="Corrected Signal", c="g")


    plt.xlabel("Time [s]")
    plt.xlim(0, (920-865))
    plt.legend(fontsize=20)
    # plt.ylabel("Signal intensity [A.U.]")
    sns.despine()
    # plt.tight_layout()
    plt.savefig("isosbestic_fp.svg")
    plt.show()

    processed_data = np.vstack((time,
                                corrected_fluorescence_signal_fiber_0,
                                corrected_fluorescence_signal_fiber_1))
    # np.save(r"Z:\aDaigle\Prediction AI\Data\Simultaneous recordings\03_18_02_2025\processed_fiber.npy", processed_data)
