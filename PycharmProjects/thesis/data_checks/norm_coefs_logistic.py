#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date of creation: 13 March 2018
Date of last edit: 04 May 2018
Script for visualisation of the feature coefficient after performing Logistic Regression

Output is .h5 file containing a compressed numpy array of feature coefficients from the logistic regression classifier
"""

import time
import os
import h5py
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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


def logistic_regression(samples, labels, samples_val, labels_val, n_samples, n_check, n_neighbours=0, title=''):
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

    # Split the data in a training (90%) and test set (10%)
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.1)

    # Keep track of the running time of the logistic regression classification
    start_time_clf = time.time()

    # Specify the classifier
    clf = linear_model.LogisticRegression()

    # Cross validated the samples on the logistic classifier
    clf, training_accuracy, training_std, test_accuracy, test_std, training_roc, training_roc_sd, test_roc, \
    test_roc_sd, accuracy_val, ROC_AUC_val = cross_validation(clf, samples, labels, samples_val, labels_val)

    # Fit the logistic regression model
    # clf.fit(X=x_train, y=y_train)

    # Get the coefficients for each feature
    # coef = clf.coef_.ravel()
    coef = []

    # # Get the training, test, and validation accuracy in percentages
    # accuracy_train, ROC_AUC_train = get_accuracy(clf, x_train, y_train)
    # accuracy_test, ROC_AUC_test = get_accuracy(clf, x_test, y_test)
    # accuracy_val, ROC_AUC_val = get_accuracy(clf, samples_val, labels_val)

    # Get the total running time of the logistic regression classification
    log_run_time = time.time() - start_time_clf

    # # Store the statistical outcomes
    # store_statistics(accuracy_train, ROC_AUC_train, accuracy_test, ROC_AUC_test, accuracy_val, ROC_AUC_val, n_samples,
    #                  n_neighbours, title, n_check, log_run_time)
    store_statistics_cv(training_accuracy, training_roc, test_accuracy, test_roc, accuracy_val, ROC_AUC_val,
                        n_samples, n_neighbours, title, n_check, log_run_time, training_std, training_roc_sd, test_std,
                        test_roc_sd)

    return coef


def cross_validation(clf, samples, labels, samples_val, labels_val):
    """ Fit the classifier model with 10-Fold cross validation and gives the test accuracy back

    clf: sklearn class, defines the classifier model
    samples: numpy array, contains features scores with shape (samples, number of features * number of neighbouring \
     positions)
    labels: numpy array, contains data labels corresponding to the samples
    """

    # Defines the K-fold
    k_fold = KFold(n_splits=10, shuffle=True)

    # Keep track of the test accuracies
    all_training_scores = []
    all_test_scores = []
    # all_val_scores = []
    all_training_roc = []
    all_test_roc = []
    # all_val_roc = []

    for train_index, test_index in k_fold.split(samples):
        # Get a training and test set for the samples and corresponding labels
        x_train, x_test = samples[train_index], samples[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # Fit the classifier and give the test accuracy
        clf.fit(X=x_train, y=y_train) #.score(X=x_test, y=y_test)
        # Get the training, test, and validation accuracy in percentages
        accuracy_train, ROC_AUC_train = get_accuracy(clf, x_train, y_train)
        accuracy_test, ROC_AUC_test = get_accuracy(clf, x_test, y_test)

        all_training_scores.append(accuracy_train)
        all_test_scores.append(accuracy_test)
        # all_val_scores.append(accuracy_val)
        all_training_roc.append(ROC_AUC_train)
        all_test_roc.append(ROC_AUC_test)
        # all_val_roc.append(ROC_AUC_val)

    accuracy_val, ROC_AUC_val = get_accuracy(clf, samples_val, labels_val)

    # Convert the test accuracy list into a numpy array
    all_training_scores = np.array(all_training_scores)
    all_test_scores = np.array(all_test_scores)
    # all_val_scores = np.array(all_val_scores)
    all_training_roc = np.array(all_training_roc)
    all_test_roc = np.array(all_test_roc)
    # all_val_roc = np.array(all_val_roc)

    training_accuracy = all_training_scores.mean()
    training_std = all_training_scores.std() * 2
    test_accuracy = all_test_scores.mean()
    test_std = all_test_scores.std() * 2
    training_roc = all_training_roc.mean()
    training_roc_sd = all_training_roc.std() * 2
    test_roc = all_test_roc.mean()
    test_roc_sd = all_test_roc.std() * 2
    # all_val_scores = all_val_scores.mean()
    # all_val_scores = all_val_scores.std() * 2

    return clf, training_accuracy, training_std, test_accuracy, test_std, training_roc, training_roc_sd, test_roc, \
           test_roc_sd, accuracy_val, ROC_AUC_val


def get_accuracy(clf, samples, labels):
    """ Get the accuracy and ROC AUC score of the logistic regression classifier

    clf: sklearn class, defines the logistic classification model
    samples: numpy array, contains features scores with shape (samples, number of features * number of neighbouring \
     positions)
    labels: numpy array, contains data labels corresponding to the samples
    """

    # Predict the labels
    predictions = clf.predict(X=samples)

    # Get the accuracy by comparing the true labels with the predicted labels
    accuracy = (accuracy_score(y_true=labels, y_pred=predictions)) * 100

    # Get the distance of the samples to the separating hyperplane
    scores = clf.decision_function(X=samples)

    # Get the ROC AUC score by comparing the true labels with the distance scores
    ROC_AUC = (roc_auc_score(y_true=labels, y_score=scores)) * 100

    return accuracy, ROC_AUC


def store_statistics(accuracy_train, ROC_AUC_train, accuracy_test, ROC_AUC_test, accuracy_val, ROC_AUC_val, n_samples,
                     n_neighbours, title, order_check, run_time):
    """ Write the statistical results to the desired .txt file

    accuracy_train: float, training accuracy after fitting the logistic regression classifier
    ROC_AUC_train: float, ROC AUC score of the training after fitting the logistic regression classifier
    accuracy_test: float, test accuracy after fitting the logistic regression classifier
    ROC_AUC_test: float, ROC AUC score of the test after fitting the logistic regression classifier
    accuracy_val: float, validation accuracy after fitting the logistic regression classifier
    ROC_AUC_val: float, ROC AUC score of the validation samples after fitting the logistic regression classifier
    n_samples: integer, correspond to the number of samples in the data set
    n_neighbours: integer, indicates how many neighbouring positions are included
    title: string, indicates if neighbouring positions are include or excluded in the data while running the classifier
    run_time: float, refers to the running time of the logistic regression classifier in seconds
    """

    # Get the file name where the statistical results will be written to
    # working_dir = os.path.dirname(os.path.abspath(__file__))
    # file_name = working_dir + "/output/norm_accuracy_checks.txt"
    file_name = "/mnt/nexenta/kersj001/results/logistic/statistical_values.txt"

    # Extend the file with the results in the corresponding data types
    with open(file_name, 'a') as output:
        output.write("\n{:d}\t{:s}\t{:d}\t{:d}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.0f}".format(
                     n_samples, title, order_check, n_neighbours, accuracy_train, accuracy_test, accuracy_val,
                     ROC_AUC_train, ROC_AUC_test, ROC_AUC_val, run_time))

    output.close()


def store_statistics_cv(accuracy_train, ROC_AUC_train, accuracy_test, ROC_AUC_test, accuracy_val, ROC_AUC_val, n_samples,
                     n_neighbours, title, order_check, run_time, training_std, training_roc_sd, test_std, test_roc_sd):

    file_name = "/mnt/nexenta/kersj001/results/logistic/statistical_values_cv.txt"

    # Extend the file with the results in the corresponding data types
    with open(file_name, 'a') as output:
        output.write("\n{:d}\t{:s}\t{:d}\t{:d}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}"
                     "\t{:.4f}\t{:.0f}".format(
                     n_samples, title, order_check, n_neighbours, accuracy_train, accuracy_test, accuracy_val,
                     ROC_AUC_train, ROC_AUC_test, ROC_AUC_val, training_std, training_roc_sd, test_std, test_roc_sd,
                     run_time))


def write_output(coefs, data_size, n_neighbours=0):
    """ Write the coefficients after running the logistic regression to a compressed numpy array in .h5 format

    coefs: numpy array of shape (number of features, ), contains the coefficients for each feature
    data_size: integer, defines the number of samples in the data
    n_neighbours: integer, indicates how many neighbours are included, default=0 (only SNP of interest)
    """

    # Get the directory for storing the file
    working_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = working_dir + "/output/CV_classifiers/clf_HDF5_files/normalized_logistic/" \
                              "{}_norm_weights_{}".format(n_neighbours, data_size)

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
    parser.add_option("-v", "--valdata", dest="validation_data", help="Path to the normalized validation samples",
        default="")
    parser.add_option("-i", "--iter", dest="iterations", help="no", default="")

    # Get the command line options for reading the data for both the benign and deleterious SNPs
    (options, args) = parser.parse_args()
    data_directory = options.data
    val_data_directory = options.validation_data
    n_check = int(options.iterations)

    return data_directory, val_data_directory, n_check


if __name__ == "__main__":

    # Keep track of the running time
    start_time_script = time.time()

    # Get the given arguments
    data_directory, val_data_directory, n_check = get_arguments()

    # Read the HDF5 file back to a numpy array
    data, data_size, n_neighbours = data_reading(data_directory)
    val_data, _, _ = data_reading(val_data_directory)

    # Parse the data into samples that considers neighbouring positions and into samples with only the SNP of interest
    snp_data, neighbour_data = data_parser(data)
    val_snp_data, val_neighbour_data = data_parser(val_data)

    # Get the data labels
    labels_data = data_labels(data)
    labels_val = data_labels(val_data)

    # Run the  logistic regression classifier for samples containing only the data of the SNP of interest and also for
    # the samples including neighbouring positional data
    # print("NEIGHBOURS EXCLUDED")
    # weights_snp = logistic_regression(snp_data, labels_data, val_snp_data, labels_val, data_size, n_check,
    #                                   title='excluding')
    print("NEIGHBOURS INCLUDED")
    weights_neighbour = logistic_regression(neighbour_data, labels_data, val_neighbour_data, labels_val, data_size,
                                            n_check, n_neighbours=n_neighbours, title='including')

    # Write the output to a HDF5 file
    # write_output(weights_snp, data_size)
    # write_output(weights_neighbour, data_size, n_neighbours=n_neighbours)

    # Get the complete running time of the script
    print("----- {} seconds -----".format(round(time.time() - start_time_script), 2))

    # Example for running from the command line
    """
    python PycharmProjects/thesis/data_checks/norm_coefs_logistic.py 
    --data /mnt/scratch/kersj001/data/output/normalized_data/5_2000000.h5 
    --valdata /mnt/scratch/kersj001/data/output/normalized_data/5_26172.h5
    """