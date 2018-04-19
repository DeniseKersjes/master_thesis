#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date of creation: 12 April 2018
Date of last edit: 18 April 2018
Script for plotting the data in a dimensional space

Output are .png files containing a PCA and MDS plot
"""

import time
import h5py
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from optparse import OptionParser


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
    """ Function that parsed the data in order to perform logistic regression with and without neighbouring positions

    data: numpy array of shape (number of samples, number of features, number of nucleotides), the data contain both
     the benign and deleterious samples
    snp_neighbour: integer, indicates if the neural network will run with only the features of the SNP of interest
     or also includes the features of the neighbouring positions
    """

    # Get only the features of the SNP of interest
    if snp_neighbour == 0:
        # The SNP of interest samples are located at the middle position of the data sequence
        index_snp = (data.shape[2] - 1) / 2  # -1 for the SNP of interest
        samples = data[:, :, int(index_snp)]

    # Get the features of the SNP of interest and neighbouring positions
    else:
        # The data should fit in a 2D array for performing neural network. The number of samples should be stay, and
        # the number of features will be the number of features times the number of nucleotides
        samples = data.reshape([data.shape[0], -1])

    return samples


def do_pca(data):
    """ Return numpy array containing object locations of the first two principal components

    data: numpy array of shape (number of samples, number of features, number of nucleotides), the data contain both
     the benign and deleterious samples
    """

    # Make an instance of the PCA model
    pca = PCA(n_components=2)

    # Fit the PCA model for classifying per feature
    pca.fit(data)
    # Get the first two principal components
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]

    # Transform the data for classifying per class label
    transformed_data = pca.fit_transform(data)

    # Check how much variance is explained by the first two components
    print(pca.explained_variance_ratio_)

    return pc1, pc2, transformed_data


def do_mds(data):
    """ Return numpy array containing object locations of pairwise distances for dimensional visualization

    data: numpy array of shape (number of samples, number of features, number of nucleotides), the data contain both
     the benign and deleterious samples
    """

    # Make an instance of the MDS model
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)

    ### Warning: calculating distances need a lot of memory since it is using .dot files
    # Calculate the distance between each point and the other points in the dataset
    class_distances = pairwise_distances(data)
    feature_distance = pairwise_distances(data.T)

    # Fit the data on the MDS model for classifying per feature
    fitted_mds_feature = mds.fit_transform(feature_distance)

    # Get the first two principal components
    pc1 = fitted_mds_feature[:, 0]
    pc2 = fitted_mds_feature[:, 1]

    # Fit the data on the MDS model for classifying per class label
    fitted_mds_class = mds.fit_transform(class_distances)

    return pc1, pc2, fitted_mds_class


def dim_class_plot(dim_data, dim_val_data, n_samples, n_neighbours, dim_type='MDS'):
    """ Saves .png figure containing scatterplot of sample values classified per class label

    dim_data: numpy array of shape (number of samples, number of dimension), containing object locations for a
     dimensional scatterplot
    dim_val_data: numpy array of shape (number of samples, number of dimension), containing validation object locations
     for a dimensional scatterplot
    n_samples: integer, correspond to the number of samples in the data set
    n_neighbours: integer, indicates how many neighbouring positions are considered
    dim_type: string, defines which dimensional classification is used, default='MDS'
    """

    # Set figure with plot size
    plt.figure(figsize=(9, 8))
    # Define the number of (validation) samples for each label class
    n_class_samples = int(dim_data.shape[0] / 2)
    n_val_class_samples = int(dim_val_data.shape[0] / 2)
    # Plot the data samples
    plt.scatter(dim_data[:n_class_samples, 0], dim_data[:n_class_samples, 1],
                label="benign", marker="o", color="xkcd:light blue")
    plt.scatter(dim_data[n_class_samples + 1:n_samples, 0], dim_data[n_class_samples + 1:n_samples, 1],
                label="deleterious", marker="o", color="xkcd:blue")
    # Plot the validation data samples
    plt.scatter(dim_val_data[:n_val_class_samples, 0], dim_val_data[:n_val_class_samples, 1],
                label="validation benign", marker="x", color="xkcd:salmon")
    plt.scatter(dim_val_data[n_val_class_samples + 1:n_samples, 0], dim_val_data[n_val_class_samples + 1:n_samples, 1],
                label="validation deleterious", marker="x", color="xkcd:brick red")
    # Set the title and axises
    plot_layout(dim_type, n_neighbours, n_samples)
    # Set the legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, ncol=4, frameon=False)
    plt.subplots_adjust(bottom=0.2)
    # Save the plot
    working_dir = os.path.dirname(os.path.abspath(__file__))
    saving(working_dir + "/output/dimension_plots/{}/class_{}_{}".format(dim_type, n_neighbours, n_samples))


def dim_feature_plot(x_pc, y_pc, x_pc_val, y_pc_val, n_samples, n_neighbours, dim_type):
    """ Saves .png figure containing scatterplot of sample values classified per feature

    x_pc: numpy array of shape (number of features, ), containing object x-value locations for a dimensional scatterplot
    y_pc: numpy array of shape (number of features, ), containing object y-value locations for a dimensional scatterplot
    x_pc_val: numpy array of shape (number of features, ), containing validation object x-value locations for a
     dimensional scatterplot
    y_pc_val: numpy array of shape (number of features, ), containing validation object y-value locations for a
     dimensional scatterplot
    n_samples: integer, correspond to the number of samples in the data set
    n_neighbours: integer, indicates how many neighbouring positions are considered
    dim_type: string, defines which dimensional classification is used
    """

    # Set figure with plot size
    plt.figure(figsize=(9, 8))
    # Get the feature names and defined colors
    feature_names, color_names = feature_colors()
    # Get the total number of features (number of feature per nucleotide position times the number of positions)
    n_features = x_pc.shape[0]
    n_positions = int(n_features / 14)  # Their are 14 features per nucleotide positions
    # Plot the dimensional values for each feature in a separate color
    for idx, start_new_feature in enumerate(range(0, n_features, n_positions)):
        range_feature_idx = range(start_new_feature, start_new_feature+n_positions)
        plt.scatter(x_pc[range_feature_idx], y_pc[range_feature_idx],
                    marker="o", label=feature_names[idx], color=color_names[idx])
        plt.scatter(x_pc_val[range_feature_idx], y_pc_val[range_feature_idx],
                    marker="x", label='validation '+feature_names[idx], color=color_names[idx])
    # Set the title and axises
    plot_layout(dim_type, n_neighbours, n_samples)
    # Set the legend
    plt.legend(loc='right', bbox_to_anchor=(1.5, 0.5), fancybox=True, frameon=False)
    plt.subplots_adjust(right=0.65)
    # Save the plot
    working_dir = os.path.dirname(os.path.abspath(__file__))
    saving(working_dir + "/output/dimension_plots/{}/feature_{}_{}".format(dim_type, n_neighbours, n_samples))


def feature_colors():
    """ Defines for each feature a name a unique color
    """

    # Define the used features in correct order
    features = ['ntA', 'ntC', 'ntT', 'ntG',
                'phastcon_mam', 'phastcon_pri', 'phastcon_ver',
                'phylop_mam', 'phylop_pri', 'phylop_ver',
                'GerpN', 'GerpS', 'GerpRS', 'Gerp_pval']

    # Define for each feature a color
    colors = ['xkcd:blue', 'xkcd:light blue', 'xkcd:emerald green', 'xkcd:soft green',
              'xkcd:beige', 'xkcd:mustard', 'xkcd:goldenrod',
              'xkcd:hot pink', 'xkcd:pink', 'xkcd:light pink',
              'xkcd:purple', 'xkcd:light purple', 'xkcd:brown', 'xkcd:light brown']

    return features, colors


def plot_layout(dim_type, n_neighbours, n_samples):
    """ Creates the layout of the plot

    dim_type: string, defines which dimensional classification is used
    n_neighbours: integer, indicates how many neighbouring positions are considered
    n_samples: integer, correspond to the number of samples in the data set
    """

    # Set the plot title
    if dim_type == 'MDS':
        plt.title('Multidimensional scaling plot\n{} samples, {} neighbours included'.format(n_samples, n_neighbours))
    elif dim_type == 'PCA':
        plt.title('Principal component analysis plot\n{} samples, {} neighbours included'.format(n_samples,
                                                                                                 n_neighbours))

    # Set axis labels
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.axis('equal')


def saving(file_path, file_type='png'):
    """ Saves the plot in the correct directory with an unique name

    file_path: string, path direction to where the plot will be saved
    file_type: string, desired format of the file, default='png'
    """

    # Get a unique file name
    index_saving = 1
    while os.path.exists(file_path + "_{}.{}".format(index_saving, file_type)):
        index_saving += 1

    # Save the figure with the unique file name
    plt.savefig(file_path + "_{}.{}".format(index_saving, file_type), bbox_inches='tight')


if __name__ == "__main__":
    # Keep track of the running time
    start_time = time.time()

    # # Get the desired parameters
    # # Define if you only  want to consider SNP positions or also neighbouring positions
    # incl_neighbour = "y"
    # # Get the data directory
    # data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/2_20000.h5"
    # val_data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/2_26172.h5"

    # Specify the options for running from the command line
    parser = OptionParser()
    # Specify the data directory for the benign and deleterious SNPs
    parser.add_option("-d", "--data", dest="data", help="Path to the output of the normalized feature scores of \
        deleterious SNPs and its neighbouring features", default="")
    # Specify the data directory for the validation samples
    parser.add_option("-v", "--valdata", dest="validation_data", help="Path to the normalized validation samples",
        default="")
    # Specify if the neural network will included neighbouring positions
    parser.add_option("-n", "--neighbours", dest="neighbours", help="String that indicates if the surrounding neighbours \
        will be included ('n') or excluded ('s')", default="n")

    # Get the command line options for reading the data for both the benign and deleterious SNPs
    (options, args) = parser.parse_args()
    data_directory = options.data
    val_data_directory = options.validation_data
    incl_neighbour = options.neighbours

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
    samples = data_parser(data=data, snp_neighbour=n_neighbours)
    val_samples = data_parser(data=val_data, snp_neighbour=n_neighbours)

    # Apply PCA on the data
    pca_first_pc, pca_second_pc, pca_pcs_label = do_pca(data=samples)
    pca_first_pc_val, pca_second_pc_val, pca_pcs_label_val = do_pca(data=val_samples)

    # Apply MDS on the data
    mds_first_pc, mds_second_pc, mds_pcs_label = do_mds(data=samples)
    mds_first_pc_val, mds_second_pc_val, mds_pcs_label_val = do_mds(data=val_samples)

    # Create a dimensional plot classified per feature
    dim_feature_plot(x_pc=pca_first_pc, y_pc=pca_second_pc, x_pc_val=pca_first_pc_val, y_pc_val=pca_second_pc_val,
                     n_samples=data_size, n_neighbours=n_neighbours, dim_type='PCA')
    dim_feature_plot(x_pc=mds_first_pc, y_pc=mds_second_pc, x_pc_val=mds_first_pc_val, y_pc_val=mds_second_pc_val,
                     n_samples=data_size, n_neighbours=n_neighbours, dim_type='MDS')

    # Create a dimensional plot classified per class label
    dim_class_plot(dim_data=pca_pcs_label, dim_val_data=pca_pcs_label_val, n_samples=data_size,
                   n_neighbours=n_neighbours, dim_type='PCA')
    dim_class_plot(dim_data=mds_pcs_label, dim_val_data=mds_pcs_label_val, n_samples=data_size,
                   n_neighbours=n_neighbours, dim_type='MDS')

    # Get the complete running time of the script
    print("----- {} seconds -----".format(round(time.time() - start_time), 2))
