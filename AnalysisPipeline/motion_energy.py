## Gracieuseté d'Antoine Daigle, 2024
from tkinter import filedialog
from tkinter import *
import cv2
from tqdm import tqdm
import numpy as np
import os


def motion_energy(video_path, avi_name):
    """Extait le motion energy du visage de la souris. Voir exemple pour tracer le ROI 
    (du coin inférieur de l'oeil jusqu'à l'etérieur de la bouche). enregistre les données dans un fichier .npy

    Args:
        video_path (_type_): chemin du dossier contenant le vidéo
        avi_name (_type_): nom du fichier vidéo
    """

    cap = cv2.VideoCapture(os.path.join(video_path, avi_name))
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    ret, frame = cap.read()
    ret = None
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    x_mot, y_mot, w_mot, h_mot = cv2.selectROI("Select motion zone", gray_frame)

    motion = np.zeros(shape=video_length)


    for i in tqdm(range(video_length)):
        # Read frame-by-frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        motion_zone = gray_frame[y_mot:y_mot+h_mot, x_mot:x_mot+w_mot]


        if i == 0:
            old_motion_zone = motion_zone.copy()

        else:
            motion[i] = cv2.mean(cv2.absdiff(motion_zone, old_motion_zone))[0]
            old_motion_zone = motion_zone.copy()


    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    np.save(os.path.join(video_path, "face_motion.npy"), motion)

    print("'face_motion.npy' saved in video folder")


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename()
    # print(os.path.split(video_path))
    video_path, avi_name = (os.path.split(video_path)[0], os.path.split(video_path)[1])
    # print(video_path)
    # print(avi_name)

    motion_energy(video_path, avi_name)
    
