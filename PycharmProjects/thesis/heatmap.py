#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date: 2 March 2018
Script for creating a heatmap of specified data values

Output is .png file containing the heatmap of specified samples
"""

import time
import h5py
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict


def get_weights(snp_or_neighbour, n_nt, data_path):
    """ Create a pandas dataframe of the specified data that will be used for creating the heatmap

    snp_or_neighbour: string, indicates if the classifier will run with only the features of the SNP of interest or \
     also includes the features of the neighbouring positions
    n_nt: integer, indicates how many neighbouring positions for both upstream and downstream of the SNP are considered
    data_path: string, directory where the data can be found
    """

    # Create a dictionary
    weight_samples = {}

    # Get directory where the numpy array of the weights are stored
    working_dir = os.path.dirname(os.path.abspath(__file__))
    path_name = working_dir + data_path
    # Get the data for every file with the correct name in the specified directory
    for file in glob.glob(path_name+'weights_{}_*'.format(snp_or_neighbour)):
        # Read the h5py file back to a numpy array
        h5f = h5py.File(file, 'r')
        # Get the correct data for every dataset varying in data size
        for size in [2000, 20000, 200000, 2000000]:
            size_name = size
            try:
                # Load the data back to a numpy array
                weights = h5f['dataset_{}'.format(size_name)][:]
                # Convert the data into a list
                weights = weights.ravel().tolist()
                # Add the data to the correct dictionary key
                if size_name in weight_samples.keys():
                    weight_samples[size_name] += [weights]
                else:
                    weight_samples.update({size_name: [weights]})
            except:
                pass

    # Sort the dictionary by data size
    my_dic = SortedDict(weight_samples)

    # Convert the dictionary into a list of tuples (data size, [[samples], [samples]])
    complete_samples = []
    for sample_name, sample_weights in my_dic.items():
        complete_samples.append((sample_name, sample_weights))

    # Get the y-labels for the heatmap, and the data values
    row_names = []
    weights = []
    for samples in complete_samples:
        # position 1 contains the sample weights and position 0 the sample size
        n_samples = len(samples[1])
        for i in range(n_samples):
            row_names.append(samples[0])
        # Get the sample weights
        for sample in samples[1]:
            weights.append(sample)

    # Get the feature labels; x-labels of the heatmap
    feature_names = feature_labels(n_nt)

    # Specify the rows and columns of the data frame
    columns = pd.Index(feature_names, name="features")
    rows = pd.Index(row_names, name="sample size")

    # Create the data frame
    df_weights_samples = pd.DataFrame(data=weights, index=rows, columns=columns)

    return df_weights_samples


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


def heatmap(weights, plot_width, subtitle, out_directory, file_name):
    """ Saves a heatmap of the weights values for every features

    weights: pandas dataframe, containing weights values for different features (columns) in combination with the data
     size (rows)
    plot_width: integer, indicates the width for the heatmap plot
    subtitle: string, subtitle of the heatmap which gives details about obtaining the sample weights
    out_directory: string, directory where the heatmap can be written to
    file_name: string, name of the heatmap file (without the file format)
    """

    # Define the width and height of the plot
    plt.subplots(figsize=(5 * plot_width, 5))
    # Create the heatmap
    sns.heatmap(weights, xticklabels=True, yticklabels=True, cmap='seismic', center=0, linewidths=.5)
    # Set the labels and title
    plt.yticks(rotation=360)  # There is a bug which cause that you have to turn the labels 360oC to get them horizontal
    plt.xlabel("Features")
    plt.ylabel("Sample size")
    plt.title('Influence of each feature\n{}'.format(subtitle), size=10)
    # Plot the figure with correct lay-out
    plt.tight_layout()

    # Save the heatmap
    working_dir = os.path.dirname(os.path.abspath(__file__))
    saving(file_path=working_dir + out_directory + file_name)


def saving(file_path, file_type='png'):
    """ Saves the plot in the correct directory with an unique name

    file_path: string, path direction to where the plot will be saved
    file_type: string, specifies the file type, default='png'
    """

    # Search for an unique file name
    index_saving = 1
    while os.path.exists(file_path + "_{}.{}".format(index_saving, file_type)):
        index_saving += 1

    # Save the figure with the unique file name
    plt.savefig(file_path + "_{}.{}".format(index_saving, file_type))


if __name__ == "__main__":
    # Keep track of the running time
    start_time = time.time()

    # Specify if you want a heatmap that considers neighbouring positions and how many
    snp_or_neighbour = "snp"
    n_nt = 0
    # snp_or_neighbour = "neighbour"
    # n_nt = 5

    # The width of the plot will in increase linearly with the number of considered neighbouring positions
    if n_nt == 0:
        plot_width = 1
    else:
        plot_width = n_nt

    # Specify the classifier
    classifier = "logistic regression"
    # classifier = "neural network"

    # Get the data directory and output variables for the defined classifier
    if classifier == "logistic regression":
        data_path = "/data_checks/output/CV_classifiers/clf_HDF5_files/"
        output_path = "/data_checks/output/CV_classifiers/clf_heatmap/"
        file_name = "heatmap_coefs_{}".format(snp_or_neighbour)
        subtitle = "Logistic Regression; trained at once on whole dataset"
    elif classifier == "neural network":
        data_path = "/output_ANN/HDF5_files/"
        output_path = "/output_ANN/heatmap/"
        file_name = "weights_multiple_{}".format(snp_or_neighbour)
        test_size = 10
        learning_rate = 0.01
        subtitle = "Neural Network; test size of {}%, learning rate of {}".format(test_size, learning_rate)

    # Get the weights of samples for making a heatmap
    weights = get_weights(snp_or_neighbour, n_nt, data_path)

    # Create the heatmap
    heatmap(weights, plot_width, subtitle, output_path, file_name)

    print("\n----- running time: {} seconds -----".format(round(time.time() - start_time), 2))