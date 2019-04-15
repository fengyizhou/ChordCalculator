import lpd_prepare_utils as utl
import numpy as np
import os
import pypianoroll as pr
import config

file = "list_chord/8.npy"

chords = np.load(file)
print(chords)