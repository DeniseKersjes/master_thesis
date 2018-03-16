#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date of creation: 2 March 2018
Date of last edit: 2 March 2018
Script for converting the feature weights to absolute weights

Output are H5py files with the compressed numpy array containing absolute feature weights
"""

import time
import h5py
import os.path
import glob
import numpy as np
from pathlib import Path


def convert_weights(data_path):
    """ Output are a h5py files containing absolute feature weight values

    data_path: string, directory that contains the desired files to be converted to absolute weights
    """

    # Get directory where the numpy array of the weights are stored
    working_dir = os.path.dirname(os.path.abspath(__file__))
    path_name = working_dir + data_path
    # Get the data for every file with the correct name in the specified directory
    for file in glob.glob(path_name + '*weights_*'):
        # Check if the file is already converted to absolute values
        file_name = file.split("/")[-1]
        absolute_file = Path(working_dir + data_path + "/absolute_values/{}".format(file_name))
        if absolute_file.is_file():
            pass
        else:
            # Check which files are converted
            print(file_name)
            # Read the h5py file back to a numpy array
            h5f = h5py.File(file, 'r')
            # The datasets are stored by size
            for size in [2000, 20000, 200000, 2000000]:
                size_name = size
                try:
                    # Load the data back to a numpy array
                    weights = h5f['dataset_{}'.format(size_name)][:]
                    # Convert the values to absolute values
                    absolute_weights = np.absolute(weights)
                    # Save the numpy array back as a h5py file
                    h5f = h5py.File(absolute_file, 'w')
                    h5f.create_dataset('dataset_{}'.format(size_name), data=absolute_weights)
                except:
                    pass


if __name__ == "__main__":
    # Keep track of the running time
    start_time = time.time()

    # Get the directory containing files with weight values
    # data_path = "/data_checks/output/CV_classifiers/clf_HDF5_files/normalized_logistic/"
    data_path = "/output_ANN/HDF5_files/normalized_ANN/"

    # Get the file names with weight values and convert them to absolute weight values
    convert_weights(data_path)

    # Get the complete running time of the script
    print("\n----- running time: {} seconds -----".format(round(time.time() - start_time), 2))