import os
import amd
import pandas as pd
import numpy as np


def compare_folders(folder_1: os.PathLike, folder_2: os.PathLike, k=100):
	path_1 = os.path.abspath(folder_1)
	path_2 = os.path.abspath(folder_2)
	dm = amd.compare(path_1, path_2, by='AMD', k=k)

	names_1 = [
	    name.strip(".cif")
	    for name in sorted(os.listdir(path_1))
	]
	names_2 = [
	    name.strip(".cif")
	    for name in sorted(os.listdir(path_2))
	]
	
	dm.index = names_1
	dm.columns = names_2

	return dm
