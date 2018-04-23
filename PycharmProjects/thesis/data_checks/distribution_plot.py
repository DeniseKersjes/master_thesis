#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date of creation: 23 April 2018
Date of last edit: 23 April 2018
Script for plotting the distribution of data features

Output are .png files containing a distribution plot
"""

import time
import h5py
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def data_reading(data_file):
    """ Reads the .H5py file containing the data back as a numpy array

    data_file: string, data directory ending with file name which contains the compressed numpy array
    """

    # The number of samples is needed to read the HDF5 file, which is stored in the name
    # Split by '/' to remove the directory, and by '.' to remove the file format
    file_name = data_file.split("/")[-1].split(".")[-2]
    # The file name ends with the number of samples and before that the number of included neighbours
    n_samples = int(file_name.split("_")[-1])
    n_neighbours = int(file_name.split("_")[-2])

    # Read the data
    h5f = h5py.File(data_file, 'r')

    # The data set name is the name of the path where the data file can be found
    data = h5f["dataset_{}".format(n_samples)][:]

    # Close the H5py file
    h5f.close()

    return data, n_samples, n_neighbours


def data_parser(data, snp_neighbour):
    """ Function that parsed the data in order to obtain a distribution plot with and without neighbouring positions

    data: numpy array of shape (number of samples, number of features, number of nucleotides), the data contain both
     the benign and deleterious samples
    snp_neighbour: integer, indicates if the neural network will run with only the features of the SNP of interest
     or also includes the features of the neighbouring positions
    """

    # Get only the features of the SNP of interest
    if snp_neighbour == 0:
        # The SNP of interest samples are located at the middle position of the data sequence
        index_snp = (data.shape[2] - 1) / 2  # -1 for the SNP of interest
        snp_samples = data[:, :, int(index_snp)]
        # Get the samples for phastcon scores (mammalian: position 4, primate: position 5, and vertebrate: position 6)
        mammalian = snp_samples[:, 4]
        primate = snp_samples[:, 5]
        vertebrate = snp_samples[:, 6]
    # Get the features of the SNP of interest and neighbouring positions
    else:
        # Get the samples for phastcon scores (mammalian: position 4, primate: position 5, and vertebrate: position 6)
        mammalian = data[:, 4, :].reshape([-1])
        primate = data[:, 5, :].reshape([-1])
        vertebrate = data[:, 6, :].reshape([-1])

    # Join the different phastcon scores together and covert it to a list
    phastcon = np.stack((mammalian, primate, vertebrate)).tolist()

    return phastcon


def distribution_plot(data, val_data, n_neighbours, n_samples):
    """

    data: numpy array of shape (number of features, number of samples * number of nucleotides), the data contain both
     the benign and deleterious samples
    val_data: numpy array of shape (number of features, number of samples * number of nucleotides), the data contain
     both the benign and deleterious validation samples
    n_neighbours: integer, indicates how many neighbouring positions are considered
    n_samples: integer, correspond to the number of samples in the data set
    """

    # Get the directory for saving the plot
    working_dir = os.path.dirname(os.path.abspath(__file__))
    save_name = working_dir + "/output/distribution_plots/PhastCon/{}_distribution_{}".format(n_neighbours, n_samples)

    # Set the samples names for the title and colors for each pahstcon type
    sample_names = ['PhastCon mammalian', 'PhastCon primate', 'PhastCon vertebrate']
    color = ['xkcd:blue', 'xkcd:emerald green', 'xkcd:red']
    # Create a density plot for each phastcon type
    for idx, phastcon_type in enumerate(data):
        # Create a new plot
        plt.figure()
        # Plot the distribution for both the training and validation set
        sns.distplot(phastcon_type, color=color[idx], label='training set')
        sns.distplot(val_data[idx], color='xkcd:medium grey', label='validation set')
        # Plot the legend
        plt.legend()
        # Set the figure and axis titles
        plt.title('Distribution of {} samples'.format(sample_names[idx]))
        plt.xlabel('Score')
        plt.ylabel('Density')
        # Save the plot
        saving(save_name, extension="-{}".format(idx))


def saving(file_path, extension='', file_type='png'):
    """ Saves the plot in the correct directory with an unique name

    file_path: string, path direction to where the plot will be saved
    extension: string, defines if the figure is a subfigure, default=''
    file_type: string, desired format of the file, default='png'
    """

    index_saving = 1
    while os.path.exists(file_path + "_{}{}.{}".format(index_saving, extension, file_type)):
        index_saving += 1

    plt.savefig(file_path + "_{}{}.{}".format(index_saving, extension, file_type), bbox_inches='tight')


if __name__ == "__main__":
    # Keep track of the running time
    start_time = time.time()

    # Get the desired parameters
    # Define if you only  want to consider SNP positions or also neighbouring positions
    incl_neighbour = "n"
    # Get the data directory
    data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/1_200000.h5"
    val_data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/1_26172.h5"

    # Read the HDF5 file back to a numpy array
    data, data_size, n_neighbours = data_reading(data_file=data_directory)
    val_data, _, _ = data_reading(data_file=val_data_directory)

    # Get the number of considered neighbouring positions
    incl_neighbour = incl_neighbour.lower()
    if incl_neighbour == "n" or incl_neighbour == "no":
        n_neighbours = 0
    else:
        n_neighbours = n_neighbours

    # Parse the data into samples which either consider neighbouring positions or not
    phastcon_scores = data_parser(data=data, snp_neighbour=n_neighbours)
    val_phastcon_scores = data_parser(data=val_data, snp_neighbour=n_neighbours)

    # Create the distribution plot
    distribution_plot(data=phastcon_scores, val_data=val_phastcon_scores, n_neighbours=n_neighbours,
                      n_samples=data_size)

    # Get the complete running time of the script
    print("----- {} seconds -----".format(round(time.time() - start_time), 2))
