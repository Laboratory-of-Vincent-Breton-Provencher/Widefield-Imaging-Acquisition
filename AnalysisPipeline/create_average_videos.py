import os
from prepData import identify_files, save_as_tiff, resample_pixel_value
from tqdm import tqdm
import numpy as np
import tifffile as tff
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

data = "LSCI"
path = r"D:\ggermain\2024-09-17_air_puffs\{}".format(data)

files_list = identify_files(path, ".tiff")


for idx_t in tqdm(range(100)):
    files = files_list[idx_t::100]
    
    for idx, file in enumerate(files):
        frame = tff.TiffFile(file).asarray()
        if idx == 0:
            num_frames = len(files)
            frame_shape = frame.shape
            stack_shape = (num_frames, frame_shape[0], frame_shape[1])
            _3d_stack = np.zeros(stack_shape, dtype=np.uint16)
            if idx_t == 0:
                mean_frames = np.zeros((100, frame_shape[0], frame_shape[1]), dtype=np.float16)
        _3d_stack[idx,:,:] = frame

    mean_frames[idx_t,:,:] = np.mean(_3d_stack, axis=0)
    # plt.imshow(mean_frames[idx_t,:,:])
    # plt.show()


# print(mean_frames.min(), mean_frames.max())

mean_frames = resample_pixel_value(mean_frames, 16)
# mean_frames = gaussian_filter(mean_frames, sigma=1)
save_as_tiff(mean_frames, "{}_mean".format(data), path)