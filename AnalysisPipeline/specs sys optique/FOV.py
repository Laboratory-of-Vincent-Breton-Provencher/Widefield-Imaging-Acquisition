import matplotlib.pyplot as plt
import numpy as np
# from scipy.signal import gaussian

from scipy.optimize import curve_fit


def my_gauss(x, s, A, b, c):

    return A * 1/(s*np.sqrt(np.pi)) * np.exp(-0.5*((x-b)**2)/(s**2)) + c


path = r"D:\ggermain\2025-03-06_FOV\profil.csv"
distance, intensite = np.loadtxt(path, skiprows=1, delimiter=',').transpose()
intensite = (intensite-intensite.min())/(intensite.max()-intensite.min())

# popt, pcov = curve_fit(my_gauss, distance, intensite, p0=(2, 70, 4, 100))

plt.plot(distance, intensite, color='grey', label='profil')
# plt.plot(distance, my_gauss(distance, *popt), label="Fit gaussien", color='k')
plt.hlines(0.2, 1.6, 7.4, color='k', linestyles='--', label="20% d'intensité")
plt.legend()
plt.ylabel("Intensité normalisée[-]")
plt.xlabel("Position [mm]")

# print(popt[0]*2.355)
plt.savefig("FOV_profil.png", dpi=600)
plt.show()