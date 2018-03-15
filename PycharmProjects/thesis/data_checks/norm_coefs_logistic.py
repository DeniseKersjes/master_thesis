#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date of creation: 13 March 2018
Date of last edit: 15 March 2018
Script for visualisation of the feature coefficient after performing Logistic Regression

Output is .h5 file containing a compressed numpy array of feature coefficients from the logistic regression classifier
"""

import time
import os
import h5py
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, accuracy_score
from optparse import OptionParser


def data_reading(data_file):
    """ Reads the .H5py file containing the data back as a nunmpy array

    data_file: string, data directory ending with file name which contains the compressed numpy array
    """

    # The number of samples is needed to read the HDF5 file, which is stored in the name
    # Split by '/' ro remove the directory, and by '.' to remove the file format
    file_name = data_file.split("/")[-1].split(".")[-2]
    # The file name ends with the number of samples
    n_samples = int(file_name.split("_")[-1])
    n_neighbours = int(file_name.split("_")[-2])

    # Read the data
    h5f = h5py.File(data_file, 'r')

    # The data set name is the name of the path where the data file can be found
    data = h5f["dataset_{}".format(n_samples)][:]

    # Close the H5py file
    h5f.close()

    return data, n_samples, n_neighbours


def data_parser(data):
    """ Function that parsed the data in order to perform logistic regression with and without neighbouring positions

    data: numpy array of shape (number of samples, number of features, number of nucleotides), the data contain both
     the benign and deleterious samples
    """

    # Get only the features of the SNP of interest, which is located at the middle position of the data
    index_SNPi = (data.shape[2] - 1) / 2  # -1 for the SNP of interest
    SNPi_samples = data[:, :, int(index_SNPi)]

    # Reshape the data sets including neighbouring SNPs to a 2D array
    # The number of samples should be stay, and the number of features will be the number of features times the number
    # of nucleotides
    neighbour_samples = data.reshape([data.shape[0], -1])

    return SNPi_samples, neighbour_samples


def data_labels(data):
    """ Get the data labels that correspond to the data samples

    data: numpy array of shape (number of samples, number of features, number of nucleotides), the data contain both
     the benign and deleterious samples
    """

    # The data consists of a equal number of benign and deleterious samples
    # The first part of the data are the benign samples (label 0), and the second part the deleterious ones (label 1)
    n_samples = data.shape[0]
    n_class_samples = int(n_samples / 2)

    # Get a numpy array of the labels
    labels_ben = [0] * n_class_samples
    labels_del = [1] * n_class_samples
    labels = np.array(labels_ben + labels_del)

    return labels


def logistic_regression(samples, labels, samples_val, labels_val, n_samples, n_neighbours=0, title=''):
    """ Perform logistic regression classification and stores the obtained feature coefficients as an HDF5 file

    samples: numpy array, contains features scores with shape (samples, number of features * number of neighbouring \
     positions)
    labels: numpy array, contains data labels corresponding to the samples
    samples_val: numpy array, contains features scores with shape (validation samples, number of features * number of \
     neighbouring positions)
    labels_val: numpy array, contains data labels corresponding to the validation samples
    n_samples: integer, correspond to the number of samples in the data set
    n_neighbours: integer, indicates how many neighbouring positions are included, default=0 (SNP of interest only)
    title: string, indicates if neighbouring positions are include or excluded in the data while running the classifier
    """

    # Specify the classifier
    clf = linear_model.LogisticRegression()

    # Fit the logistic regression model
    clf.fit(X=samples, y=labels)

    # Get the coefficients for each feature
    coef = clf.coef_.ravel()

    # Get the training accuracy in percentages
    predictions_train = clf.predict(X=samples)
    accuracy_train = (accuracy_score(y_true=labels, y_pred=predictions_train))*100
    scores_train = clf.decision_function(X=samples)
    ROC_AUC_train = (roc_auc_score(y_true=labels, y_score=scores_train))*100

    # Check the validation accuracy in percentages
    predictions_val = clf.predict(X=samples_val)
    accuracy_val = (accuracy_score(y_true=labels_val, y_pred=predictions_val))*100
    scores_val = clf.decision_function(X=samples_val)
    ROC_AUC_val = (roc_auc_score(y_true=labels_val, y_score=scores_val))*100

    # Store the statistical outcomes
    store_statistics(accuracy_train, ROC_AUC_train, accuracy_val, ROC_AUC_val, n_samples, n_neighbours, title)

    return coef


def store_statistics(accuracy_train, ROC_AUC_train, accuracy_val, ROC_AUC_val, n_samples, n_neighbours, title):
    """ Write the statistical results to the desired .txt file

    accuracy_train: float, training accuracy after fitting the logistic regression classifier
    ROC_AUC_train: float, ROC AUC score of the training after fitting the logistic regression classifier
    accuracy_val: float, validation accuracy after fitting the logistic regression classifier
    ROC_AUC_val: float, ROC AUC score of the validation samples after fitting the logistic regression classifier
    n_samples: integer, correspond to the number of samples in the data set
    n_neighbours: integer, indicates how many neighbouring positions are included
    title: string, indicates if neighbouring positions are include or excluded in the data while running the classifier
    """

    # Get the file name where the statistical results will be written to
    working_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = working_dir + "/output/norm_accuracy.txt"

    # Extend the file with the results in the corresponding data types
    with open(file_name, 'a') as output:
        output.write("\n{:d}\t{:s}\t{:d}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(n_samples, title, n_neighbours,
                     accuracy_train, accuracy_val, ROC_AUC_train, ROC_AUC_val))

    output.close()


def write_output(coefs, data_size, n_neighbours=0):
    """ Write the coefficients after running the logistic regression to a compressed numpy array in .h5 format

    coefs: numpy array of shape (number of features, ), contains the coefficients for each feature
    data_size: integer, defines the number of samples in the data
    n_neighbours: integer, indicates how many neighbours are included, default=0 (only SNP of interest)
    """

    # Get the directory for storing the file
    working_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = working_dir + "/output/CV_classifiers/clf_HDF5_files/normalized_logistic/{}_norm_weights_{}".format(
        n_neighbours, data_size)

    # Get an unique file name
    file_name = saving(file_name)

    # Save the numpy array containing the coefficients as an HDF5 file
    h5f = h5py.File(file_name, 'w')
    h5f.create_dataset('dataset_{}'.format(data_size), data=coefs)


def saving(file_path, format_type="h5"):
    """ Get an unique name for the .h5 file

    file_path: string, directory where the file can be written to
    format_type: string, type of the created file, default="h5"
    """

    # Get an unique file name
    index_saving = 1
    while os.path.exists(file_path+"_{}.{}".format(index_saving, format_type)):
        index_saving += 1

    file_name = file_path+"_{}.{}".format(index_saving, format_type)

    return file_name


def get_arguments():
    """ Return the arguments given on the command line
    """

    # # If you do not run from the command line
    # data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/5_200000.h5"
    # val_data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/5_26172.h5"

    # Read the data
    # Specify the options for running from the command line
    parser = OptionParser()
    # Specify the data directory for the benign and deleterious SNPs
    parser.add_option("-d", "--data", dest="data", help="Path to the output of the normalized feature scores of \
        deleterious SNPs and its neighbouring features", default="")
    # Specify the data directory for the validation samples
    parser.add_option("-v", "--valdata", dest="validation_data", help="Path to the normalized validation samples", \
        default="")

    # Get the command line options for reading the data for both the benign and deleterious SNPs
    (options, args) = parser.parse_args()
    data_directory = options.data
    val_data_directory = options.validation_data

    return data_directory, val_data_directory


if __name__ == "__main__":

    # Keep track of the running time
    start_time_script = time.time()

    # Get the given arguments
    data_directory, val_data_directory = get_arguments()

    # Read the HDF5 file back to a numpy array
    data, data_size, n_neighbours = data_reading(data_directory)
    val_data, _, _ = data_reading(val_data_directory)

    # Convert the NaN data values into zero's
    data = np.nan_to_num(data)
    val_data = np.nan_to_num(val_data)

    # Parse the data into samples that considers neighbouring positions and into samples with only the SNP of interest
    snp_data, neighbour_data = data_parser(data)
    val_snp_data, val_neighbour_data = data_parser(val_data)

    # Get the data labels
    labels_data = data_labels(data)
    labels_val = data_labels(val_data)

    # Run the  logistic regression classifier for samples containing only the data of the SNP of interest and also for
    # the samples including neighbouring positional data
    print("NEIGHBOURS EXCLUDED")
    weights_snp = logistic_regression(snp_data, labels_data, val_snp_data, labels_val, data_size, title='excluding')
    print("NEIGHBOURS INCLUDED")
    weights_neighbour = logistic_regression(neighbour_data, labels_data, val_neighbour_data, labels_val, data_size,
                                            n_neighbours=n_neighbours, title='including')

    # Write the output to a HDF5 file
    write_output(weights_snp, data_size)
    write_output(weights_neighbour, data_size, n_neighbours=n_neighbours)

    # Get the complete running time of the script
    print("----- {} seconds -----".format(round(time.time() - start_time_script), 2))

    # Example for running from the command line
    """
    python PycharmProjects/thesis/data_checks/norm_coefs_logistic.py 
    --data /mnt/scratch/kersj001/data/output/normalized_data/5_2000000.h5 
    --valdata /mnt/scratch/kersj001/data/output/normalized_data/5_26172.h5
    """