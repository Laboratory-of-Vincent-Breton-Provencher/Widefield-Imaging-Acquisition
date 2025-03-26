import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

data = np.loadtxt(r"D:\ggermain\2025-03-21_opto_M914\1_whiskerpad\test_freq1.csv", skiprows=1, delimiter=",")
data = data[:, 1]

# peaks, _ = find_peaks(data)
# # print(peaks)
# # print(peaks)
# periode = peaks[1:] - peaks[:-1]
# print(np.mean(periode))

# plt.plot(data)
# plt.vlines(peaks, min(data), max(data), colors='k')
# plt.show()

datafft = np.fft.fft(data)
freq = np.fft.fftfreq(data.shape[-1])


# # print(datafft.shape)
plt.plot(freq, datafft)
plt.ylim(-1000, 1000)
plt.show()