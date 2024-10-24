import os
from tkinter import filedialog, simpledialog
# from tkinter import *
import tkinter as tk
import shutil
from tqdm import tqdm


FLAG405 = 1
FLAG470 = 1
FLAG530 = 1
FLAG625 = 1
FLAG785 = 1

FLAGS = {"FLAG405":FLAG405, "FLAG470":FLAG470, "FLAG530":FLAG530, "FLAG625":FLAG625, "FLAG785":FLAG785}

#%%  Channels Selection Window

# window = tk.Tk()
# window.withdraw()

# class ChannelSelectionWindow(simpledialog.Dialog):
#     def __init__(self, parent, title=None):
#         self.result = None
#         super().__init__(parent, title=title)

#     def body(self, frame):
#         tk.Label(frame, text='Select active channels', height=2, font=("Arial", 12)).grid(row=0)

#         self.var405 = tk.IntVar(value=1)
#         self.c405 = tk.Checkbutton(frame, text='405 nm (GCaMP isosbestic)',variable=self.var405, onvalue=1, offvalue=0, ).grid(row=1, sticky='W')

#         self.var470 = tk.IntVar(value=1)
#         self.c470 = tk.Checkbutton(frame, text='470 nm (GCaMP excitation)',variable=self.var470, onvalue=1, offvalue=0).grid(row=2, sticky='W')

#         self.var530 = tk.IntVar(value=1)
#         self.c530 = tk.Checkbutton(frame, text='530 nm (IOI isosbestic)',variable=self.var530, onvalue=1, offvalue=0).grid(row=3, sticky='W')

#         self.var625 = tk.IntVar(value=1)
#         self.c625 = tk.Checkbutton(frame, text='625 nm (IOI)',variable=self.var625, onvalue=1, offvalue=0).grid(row=4, sticky='W')

#         self.var785 = tk.IntVar(value=1)
#         self.c785 = tk.Checkbutton(frame, text='785 nm (LSCI)',variable=self.var785, onvalue=1, offvalue=0).grid(row=5, sticky='W')

#         # return self.var405, self.var470, self.var530, self.var625, self.var785

# test = ChannelSelectionWindow(window)

#%%

root = tk.Tk()
root.withdraw()
folderPath = filedialog.askdirectory()

# folderPath = r"C:\Users\gabri\Desktop\testAnalyse"


print("Sorting data")

filesPaths = [os.path.join(folderPath, file) for file in os.listdir(folderPath)]
filesPaths.sort(key=lambda x: os.path.getmtime(x))


j = 0
for flagName, flag in zip(FLAGS.keys(), FLAGS.values()):
    if flag:
        try:
            os.makedirs(os.path.join(folderPath, flagName[4:7]))
            print("Directory {} was created successfully".format(flagName[4:7]))
        except OSError as error:
            print("Directory {} already exists".format(flagName[4:7]))
        filesChannel = filesPaths[j::sum(FLAGS.values())]

        for file in tqdm(filesChannel):
            splitPath = os.path.split(file)
            shutil.move(file, os.path.join(splitPath[0], flagName[4:7], splitPath[1]))
        j += 1


# if __name__ == '__main__':
#     window.mainloop()