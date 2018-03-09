#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date: 9 March 2018
Script for normalization of the data

Output are H5py files with the compressed numpy array containing normalized feature scores
"""

import time
import h5py
import glob
import pandas as pd
import numpy as np


def data_reading(data_directory):
    """ Returns the converted H5py data set into numpy array

    data_directory: string, defined path where the data file ending with .h5 can be found
    """

    file_directory = data_directory + '*.h5'

    # Check if there is only one file with the correct name in the data directory
    n_file = 0
    for file in glob.glob(file_directory):
        n_file += 1
        if n_file > 1:
            raise SyntaxError("There are more than 1 file in the data directory that can be converted, or the data is"
                              "already converted to normalized samples")
        # Get the correct data directory including the file name
        data_directory = file
        print(data_directory)

    # Read the data
    h5f = h5py.File(data_directory, 'r')

    # The data set name is the name of the path where the data file can be found
    data = h5f[data_directory.split('/')[-2]][:]

    # Close the H5py file
    h5f.close()

    return data


def feature_labels(n_nt_downstream):
    """ Return a numpy list containing the names of the features in string format

    n_nt_downstream: integer, indicates how many neighbouring nucleotides for the downstream position of the SNP of
     interest are considered
    """

    # Define the feature labels in correct order
    feature_names = ['ntA', 'ntC', 'ntT', 'ntG',
                     'phastcon_mam', 'phastcon_pri', 'phastcon_verp',
                     'phylop_mam', 'phylop_pri', 'phylop_verp',
                     'GerpN', 'GerpS', 'GerpRS', 'Gerp_pval']

    # Get the feature names when neighbouring positions are not included
    if n_nt_downstream == 0:
        all_feature_names = np.array(feature_names)
    # Get the feature names when neighbouring positions are included
    else:
        # Get the number of neighbouring nucleotides
        n_nt_upstream = -n_nt_downstream
        # Get the downstream and upstream feature names
        all_feature_names = []
        # Add the nucleotide positions to every feature name
        for feature in feature_names:
            for i in range(n_nt_upstream, n_nt_downstream+1):  # +1 for including the SNP of interest
                # Adapt the feature name with position index behind it
                tmp = feature + ' {}'.format(i)
                all_feature_names.append(tmp)
        # Convert the feature list into a numpy array
        all_feature_names = np.array(all_feature_names)

    return all_feature_names


def create_dataframe(data):
    """ Return a dataframe of the numpy array with the features as columns and the samples as rows

    data: numpy  array with shape (number of samples, number of features, number of nucleotides)
    """

    # Reshape data to 2D array
    data_2D = data.reshape(data.shape[0], -1)

    # Get the number of consider neighbouring nucleotides
    n_nt = int((data.shape[2] - 1) / 2)  # -1 because of the SNP of interest
    # Get the feature labels; x-labels of the heatmap
    feature_names = feature_labels(n_nt)

    # Get the number of samples; y-labels of the heatmap
    n_samples = data.shape[0]
    row_names = range(1, n_samples+1)  # +1 for starting from position 1 instead of 0

    # Specify the rows and columns of the data frame
    columns = pd.Index(feature_names, name="features")
    rows = pd.Index(row_names, name="sample size")

    # Create the data frame
    dataframe = pd.DataFrame(data=data_2D, index=rows, columns=columns)

    return dataframe


def normalize_data(dataframe, data_shape, data_directory, file_name):
    """ Normalized the data and stored the dataframe in its original numpy array format in the chosen directory

    dataframe: pandas dataframe, containing the features as columns and samples as rows
    data_shape: tuple of (number of samples, number of features, number of nucleotides)
    data_directory: string, defined path where the data can be found and should be written to
    file_name: string, desired file name for the normalized numpy array
    """

    # Normalize the data by using the standard score; formula: (X - mean) / std
    normalized_df = (dataframe - dataframe.mean()) / dataframe.std()

    # Convert the dataframe back to its original numpy array shape
    numpy_values = normalized_df.values
    reshaped_numpy = numpy_values.reshape(data_shape)

    # Store the normalized dataframe as a numpy array in the defined data directory
    # Create a file in the chosen output directory
    h5f = h5py.File(data_directory + "{}.h5".format(file_name), 'w')

    # Write the data to H5py
    # The name of the data set file is equal to the name of the path where the data is coming from
    write_output = h5f.create_dataset(data_directory.split('/')[-2], data=reshaped_numpy)

    # Check if the data is written properly
    print(write_output)

    # Close the H5py file
    h5f.close()


if __name__ == "__main__":
    # Keep track of the running time
    start_time = time.time()

    # Read the data
    data_directory = "/mnt/scratch/kersj001/data/output/1_thousand_ben/combined_ben.h5"
    data = data_reading(data_directory)

    # Convert the data into a pandas dataframe
    data_df = create_dataframe(data)

    # Normalize the data and store it in its original shape in the chosen directory
    data_shape = data.shape
    file_name = "normalized"
    normalize_data(data_df, data_shape, data_directory, file_name)

    print("\n----- running time: {} seconds -----".format(round(time.time() - start_time), 2))
