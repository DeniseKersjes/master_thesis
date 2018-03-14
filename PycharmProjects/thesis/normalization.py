#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date: 13 March 2018
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
            raise SyntaxError("There are more than 1 file in the data directory to be converted")
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
                     'phastcon_mam', 'phastcon_pri', 'phastcon_ver',
                     'phylop_mam', 'phylop_pri', 'phylop_ver',
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


def create_dataframe(data_ben, data_del):
    """ Return a dataframe of the numpy array with the features as columns and the samples as rows

    data_del: numpy  array with shape (number of samples, number of features, number of nucleotides), containing the
     deleterious SNPs
    data_ben: numpy  array with shape (number of samples, number of features, number of nucleotides), containing the
     benign SNPs
    """

    # Combine the benign and deleterious SNPs to 1 array
    data = np.concatenate((data_ben, data_del), axis=0)  # 0 to put the arrays behind each other

    # Get the shape of the numpy array to convert later the dataframe back to a numpy array
    data_shape = data.shape

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
    columns = pd.Index(feature_names, name="feature")
    rows = pd.Index(row_names, name="sample number")

    # Create the data frame
    dataframe = pd.DataFrame(data=data_2D, index=rows, columns=columns)

    return dataframe, data_shape


def handle_nan(dataframe):
    """ Convert the missing values as mentioned in the appendix of CADD paper (https://www.nature.com/articles/ng.2892)

    dataframe: pandas dataframe, containing the features as columns and samples as rows
    """

    # Create a dictionary that indicates per feature how to impute the missing values
    # Values are coming from the paper 'Supplementary Information for A general framework for estimating the relative
    # pathogenicity of human genetic variants'
    impute_values = {'ntA': 0.000, 'ntC': 0.000, 'ntT': 0.000, 'ntG': 0.000,
                     'phastcon_mam': 0.079, 'phastcon_pri': 0.115, 'phastcon_ver': 0.094,
                     'phylop_mam': -0.038, 'phylop_pri': -0.033, 'phylop_ver': 0.017,
                     'GerpN': 1.909, 'GerpS': -0.200, 'GerpRS': 0.000, 'Gerp_pval': 1.000}

    no_nan_df = dataframe.copy()
    for feature_name in list(dataframe):
        # Get the feature name without position
        name = feature_name.split(' ')[0]
        # Get the correct impute value
        impute_value = impute_values[name]
        # Replace the missing values with the impute value
        no_nan_df[feature_name] = no_nan_df[feature_name].fillna(impute_value)

    return no_nan_df


def normalize_data(dataframe, data_shape):
    """ Normalized the data and convert the dataframe back in its original numpy array format

    dataframe: pandas dataframe, containing the features as columns and samples as rows
    data_shape: tuple of (number of samples, number of features, number of nucleotides)
    """

    # Normalize the data by using the standard score; formula: (X - mean) / std
    normalized_df = dataframe.copy()
    for feature_name in list(dataframe):
        # The nucleotide features should not be normalized
        if "nt" not in feature_name:
            normalized_df[feature_name] = (dataframe[feature_name] - dataframe[feature_name].mean()) / \
                                          dataframe[feature_name].std()

    # Convert the dataframe back to its original numpy array shape
    numpy_values = normalized_df.values
    reshaped_numpy = numpy_values.reshape(data_shape)

    return reshaped_numpy


def write_output(data_numpy, out_directory, data_directory):
    """ Writing the nunmpy array containing the data to H5py format

    data_numpy: numpy array of shape (number of samples, number of feature, number of nucleotides), contains both the
     deleterious and benign samples
    out_directory: string, defined path where the output should be stores
    data_directory: string, defined path where the data can be found
    """

    # The file name will consist of the number of considered neighbours and the sample amount
    n_neighbours = data_directory.split("/")[-3]
    n_samples = data_numpy.shape[0]
    file_name = "{}_{}.h5".format(n_neighbours, n_samples)

    # Create a HDF5 file in the chosen output directory
    h5f = h5py.File(out_directory + file_name, 'w')
    # Write the data to H5py
    h5f_output = h5f.create_dataset("dataset_{}".format(n_samples), data=data_numpy)
    # Check if the data is written properly
    print(h5f_output)
    # Close the H5py file
    h5f.close()


if __name__ == "__main__":
    # Keep track of the running time
    start_time = time.time()

    # Get the data directory
    data_directory_ben = "/mnt/scratch/kersj001/data/output/10/1_mil_ben/"
    data_directory_del = "/mnt/scratch/kersj001/data/output/10/1_mil_del/"

    # Read the data
    data_ben = data_reading(data_directory_ben)
    data_del = data_reading(data_directory_del)

    # Convert the data into a pandas dataframe
    data_df, data_shape = create_dataframe(data_ben, data_del)

    # Check for NaN values and handle those
    df_without_nan = handle_nan(data_df)

    # Normalize the data
    normalized = normalize_data(df_without_nan, data_shape)

    # Write the normalized data to the desired output directory
    output_directory = "/mnt/scratch/kersj001/data/output/normalized_data/"
    write_output(normalized, output_directory, data_directory_ben)

    print("\n----- running time: {} seconds -----".format(round(time.time() - start_time), 2))
