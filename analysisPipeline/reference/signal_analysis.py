"""Class used to analyse fiber photometry data."""
from collections import Counter
import itertools
import numpy as np
from scipy import interpolate
from scipy.signal import correlate, correlation_lags, medfilt
import matplotlib.pyplot as plt
import pandas as pd

class SignalAnalysis:
    """Class used to analyse signals."""

    def __init__(self) -> None:
        """Class used to analyse signals. Use for cross-correlation."""


    def __reshape_signal__(self, time_stamp:np.array, signal:np.array):
        """Method used to reshape the signal by creating a function.

        Args:
            time_stamp (np.array): Time values of the signal array (or list).
            signal (np.array): Signal array (or list).

        Returns:
            interp1d object: Function of the signal.
        """
        return interpolate.interp1d(time_stamp, signal)


    def signals_lag(self, a:list, b:list, figure=False) -> int:
        """Function that finds the lag between two signals. Code taken from 
        
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlation_lags.html#scipy.signal.correlation_lags

        Args:
            a (list): First signal (y-values).
            b (list): Second signal (y-values).
            figure (bool): If the correlation plot appears.

        Returns:
            int: Lag in position between the signals. If the lag is negative,
            the second signal is in advance on the first one. If the lag is
            positive, the first signal is in advance on the second one.
        """

        correlation = correlate(a, b, mode="full")
        lags = correlation_lags(len(a), len(b), mode="full")
        lag = lags[np.argmax(correlation)]
        if figure is True:
            plt.plot(lags, correlation)
            plt.axvline(lag)
            plt.show()
        return lag


    def resample_dataset(self, dataset:list) -> list:
        """Method that resample the dataset by linear interpolation between the
        point. This is a necessary step for cross-correlation analysis.

        Args:
            dataset (list): List with the datasets. The format is:
            [[time1, data1],
             [time2, data2],
             [time3, data3],
             ...]

        Returns:
            list: New dataset [time, data1, data2, ...].
        """

        time_list = []
        for i in dataset:
            time_list = time_list + i[0]

        time_list = sorted(Counter(time_list).keys())


        new_dataset = []
        new_data = []

        for i in dataset:
            data_function = self.__reshape_signal__(i[0], i[1])
            new_time = []
            new_data = []
            for _ in time_list:
                try:
                    new_data.append(float(data_function(_)))
                    new_time.append(_)

                except ValueError:
                    pass

            new_dataset.append([new_time, new_data])

        return new_dataset
    

    def evaluate_lag_distribution(self, lag:int, time_stamps:list,figure=False):
        """Method that calculate the lag between the signals with statistic related to those signals.The need to do some stats on the time stamps distribution comes from the fact that after the resampling, the time stamps are not perfectly separated.

        Args:
            lag (int): Lag in increment between the signals
            time_stamps (list): Time stamps
            figure (bool, optional): Appears a boxplot of the distribution. Defaults to False.

        Returns:
            tuple: mean, median, standard_deviation , confidence_intervals
        """

        time_var = []
        for i in enumerate(time_stamps):
            time_var.append(time_stamps[i[0]] - time_stamps[i[0]-abs(lag)])

        time_distribution = time_var[abs(lag):-1*abs(lag)]

        mean = np.nanmean(time_distribution)
        median = np.nanmedian(time_distribution)
        std = np.nanstd(time_distribution)
        ci = np.nanpercentile(time_distribution,[2.5, 97.5])

        if figure is True:
            plt.boxplot(time_distribution, vert=False, notch=True)
            plt.show()

        return mean, median, std, ci


    def lag_matrix(self, dataset:list) -> dict:
        """Method that analyse all the signals between each other. The cross-correlation is done with the first signal as the row and the second signal as the column. The statistical results are returned in a dictionnary whose keys are: lag, mean, median, standard_deviation, confidence_interval_lower and confidence_interval_higher

        Args:
            dataset (list): Dataset to analyse and resample. Must be in the format [[Time1, Signal1], [Time2, Signal2], ...]

        Returns:
            dict: Dictionnary with the analysed data.
        """

        resampled_dataset = self.resample_dataset(dataset)

        num_of_signals = len(resampled_dataset)

        lag_array = np.zeros((num_of_signals, num_of_signals))
        mean_array = np.zeros((num_of_signals, num_of_signals))
        median_array = np.zeros((num_of_signals, num_of_signals))
        std_array = np.zeros((num_of_signals, num_of_signals))
        ci_lower_array = np.zeros((num_of_signals, num_of_signals))
        ci_higher_array = np.zeros((num_of_signals, num_of_signals))



        for i in list(itertools.product(range(num_of_signals), repeat=2)):
            signals_lag = self.signals_lag(resampled_dataset[i[0]][1],
                                           resampled_dataset[i[1]][1])

            if signals_lag == 0:
                lag_array[i] = 0
                mean_array[i] = 0
                median_array[i] = 0
                std_array[i] = 0
                ci_lower_array[i] = 0
                ci_higher_array[i] = 0
            else:
                mean, median,std, ci = self.evaluate_lag_distribution(signals_lag, resampled_dataset[i[0]][0])

                lag_array[i] = signals_lag
                mean_array[i] = mean
                median_array[i] = median
                std_array[i] = std
                ci_lower_array[i] = ci[0]
                ci_higher_array[i] = ci[1]

        stats_dict = {
            "lag":lag_array,
            "mean":mean_array,
            "median":median_array,
            "standard_deviation":std_array,
            "confidence_interval_lower": ci_lower_array,
            "confidence_interval_higher":ci_higher_array
        }

        return stats_dict


    def median_filter(self, signal:list, window:int, figure:bool=False):
        """Method that apply a median filter to the data.

        Args:
            signal (list): Data to filter.
            window (int): Size of the kernel.
            figure (bool, optional): Plot the figure for rapid kernel testing.
             Defaults to False.

        Returns:
            list: Filtered data
        """
        filtered_data = medfilt(signal, window)

        if figure is True:
            plt.plot(signal, c="b", alpha=0.2, label="Signal")
            plt.plot(filtered_data, c="b", label="Filtered signal")
            plt.legend()
            plt.show()
        return filtered_data



if __name__ == "__main__":

    def test1():
        df = pd.read_csv(r"test_samples/test_data_1.csv")

        plt.plot(df["Time1"], df["Signal1"])
        plt.plot(df["Time2"], df["Signal2"])


        time1 = list(df["Time1"].dropna())
        signal1 = list(df["Signal1"].dropna())
        time2 = list(df["Time2"].dropna())
        signal2 = list(df["Signal2"].dropna())

        data = [[time1, signal1], [time2, signal2]]

        stat_data = SignalAnalysis().lag_matrix(data)
        print("Full cross-correlations between all signals")
        print("Lag array\n", stat_data["lag"])
        print("Mean array\n", stat_data["mean"])
        print("Median array\n", stat_data["median"])
        print("STD array\n", stat_data["standard_deviation"])
        print("CI lower array\n", stat_data["confidence_interval_lower"])
        print("CI higher array\n", stat_data["confidence_interval_higher"])


        # This section is for a precise signal.
        resampled_dataset = SignalAnalysis().resample_dataset(data)

        resampled_time1 = resampled_dataset[0][0]
        resampled_signal1 = resampled_dataset[0][1]

        resampled_time2 = resampled_dataset[1][0]
        resampled_signal2 = resampled_dataset[1][1]

        plt.scatter(resampled_time1, resampled_signal1)
        plt.scatter(resampled_time2, resampled_signal2)
        plt.show()

        signal_lag = SignalAnalysis().signals_lag(resampled_signal2, resampled_signal1)

        print("Lag", signal_lag)

        SignalAnalysis().evaluate_lag_distribution(signal_lag, resampled_time2,
                                                figure=True)

    test1()