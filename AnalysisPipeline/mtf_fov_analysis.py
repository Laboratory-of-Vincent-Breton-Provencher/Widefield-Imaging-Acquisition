import matplotlib.pyplot as plt
import tifffile as tff
import cv2
from scipy.special import erf
from scipy.optimize import curve_fit
import numpy as np



#%% --- MTF ---



# def my_erf(x, A, b, c, d):
#     return A * erf(b*(x-c)) + d



# path = r"D:\ggermain\2025-04-10_papier_fov_mtf\mtf_24mm.tiff"
# frame = tff.TiffFile(path).asarray()


# frame = frame[726:1722, 1073:2232]        # 24 mm
# # frame = frame[726:1556, 1257:2152]        # 10 mm


# plt.imshow(frame)
# # plt.savefig("zone_10mm.png", dpi=600)
# plt.show()



# x = np.linspace(1, len(frame[0]), len(frame[0]))

# # erf_array = np.zeros((frame.shape))
# for idx, line in enumerate(frame):
#     popt, pcov = curve_fit(my_erf, x, line, p0=(-2000, 0.0001, 400, 2000))
#     # plt.plot(x-popt[2], line)
#     plt.plot(x[:-1]-popt[2], -1*(my_erf(x, *popt)[1:] - my_erf(x, *popt)[:-1]) )
    
# #     erf_array[idx, :] = my_erf(x, *popt)
# plt.xlim(-30, 30)
# # plt.savefig("psf_10mm.png", dpi=600)
# plt.show()

# plt.imshow(erf_array)
# plt.show()

#%% --- FOV ---

# path = r"D:\ggermain\2025-04-10_papier_fov_mtf\fov_white.tiff"
# frame = tff.TiffFile(path).asarray()
# plt.imshow(frame)
# plt.show()

def my_erf(x, A, b, c, d):
    return A * erf(b*(x-c)) + d

path = r"D:\ggermain\2025-04-10_papier_fov_mtf\fov_mm.csv"
dist, intens = np.loadtxt(path, skiprows=1, delimiter=',').transpose()

popt1, pcov1 = curve_fit(my_erf, dist[:len(dist)//2], intens[:len(dist)//2], p0=(3000, 0.1, 5, 1500))
popt2, pcov2 = curve_fit(my_erf, dist[len(dist)//2:], intens[len(dist)//2:], p0=(-3000, 0.1, 11, 1500))

plt.plot(dist, intens, label="Profil d'intensité", color='grey')
plt.plot(dist[:len(dist)//2], my_erf(dist[:len(dist)//2], *popt1), label="fit d'une fonction d'erreur", color='k')
plt.plot(dist[len(dist)//2:], my_erf(dist[len(dist)//2:], *popt2), color='k')
plt.xlabel("Position (mm)")
plt.ylabel("Intensité (a.u.)")
plt.legend(loc=1)
# plt.savefig("FOV_profil.svg")
plt.show()

# print(popt1, popt2)
print("FOV: ", popt2[2]-popt1[2])
print((np.sqrt(pcov1[2,2])+np.sqrt(pcov2[2,2]))*2)