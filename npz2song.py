import pypianoroll as pp
import numpy as np
import copy
import os

def traverse_dir(root_dir, extension = '.npz'):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        print(root)
        for file in files:
            if file.endswith(extension):
                print(os.path.join(root, file))
                file_list.append(os.path.join(root, file))

    return file_list


file_list = traverse_dir("lpd_5_cleansed")
for file_num, file in enumerate(file_list):
    file_name = file
    file = pp.load(file_name)
    print("MIDI" + str(file_num) + "  Complete!!!")
    pp.write(file, "RES/MIDI" + str(file_num))

# file_name = "RES/All_Chord.npy"
# file = np.load(file_name)
# print(file.shape)