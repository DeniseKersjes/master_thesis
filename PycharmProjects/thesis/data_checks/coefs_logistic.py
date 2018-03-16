#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date: 2 March 2018
Script for visualisation of the feature coefficient after performing Logistic Regression

Output is .h5 file containing a compressed numpy array of feature coefficients from the logistic regression classifier
"""

import time
import os
import h5py
import numpy as np
from sklearn import linear_model
from optparse import OptionParser
from sklearn.metrics import roc_auc_score
from classifiers import data_reshape, samples_labels, data_reading, saving


def data_parser(data_ben, data_del, data_ben_val, data_del_val):
    """ Function that parsed the data in order to perform logistic regression with and without neighbouring positions

    data_ben: numpy array with shape (number of samples, number of features, number of nucleotides), the samples are\
     benign SNPs
    data_del: numpy array with shape (number of samples, number of features, number of nucleotides), the samples are\
     deleterious SNPs
     data_ben_val: numpy array with shape (number of samples, number of features, number of nucleotides), the samples \
     are validation benign samples obtained from the Exon Sequencing Project
    data_del_val: numpy array with shape (number of samples, number of features, number of nucleotides), the samples \
     are validation deleterious samples obtained from the ClinVar database
    """

    # Reshape the data into a 2D numpy array containing either only the samples of the SNP of interest or samples
    # including neighbouring positions as well
    SNPi_ben, SNPi_del, data_ben_2D, data_del_2D = data_reshape(data_ben, data_del)
    SNPi_ben_val, SNPi_del_val, data_ben_2D_val, data_del_2D_val = data_reshape(data_ben_val, data_del_val)

    # Get the samples and labels for both the training data
    samples_SNPi, labels_SNPi = samples_labels(SNPi_ben, SNPi_del)
    samples_neighbouring, labels_neighbouring = samples_labels(data_ben_2D, data_del_2D)
    samples_SNPi_val, labels_SNPi_val = samples_labels(SNPi_ben_val, SNPi_del_val)
    samples_neighbouring_val, labels_neighbouring_val = samples_labels(data_ben_2D_val, data_del_2D_val)

    return samples_SNPi, labels_SNPi, samples_neighbouring, labels_neighbouring, \
           samples_SNPi_val, labels_SNPi_val, samples_neighbouring_val, labels_neighbouring_val


def logistic_regression(samples, labels, samples_val, labels_val, data_size, title):
    """ Perform logistic regression classification and stores the obtained feature coefficients as an HDF5 file

    samples: numpy array, contains features scores with shape (samples, number of features * number of neighbouring \
     positions)
    labels: numpy array, contains data labels corresponding to the samples
    samples_val: numpy array, contains features scores with shape (validation samples, number of features * number of \
     neighbouring positions)
    labels_val: numpy array, contains data labels corresponding to the validation samples
    title: string, indicates if the classifier was run with only the features of the SNP of interest or also includes \
     the features of the neighbouring positions
    data_size: integer, defines the number of samples in the data
    """

    # Specify the classifier
    clf = linear_model.LogisticRegression()

    # Fit the logistic regression model
    clf.fit(X=samples, y=labels)

    # Get the coefficients for each feature
    coef = clf.coef_.ravel()

    # Check the accuracy of the classifier
    predictions_scores = clf.decision_function(X=samples_val)
    ROC_AUC = roc_auc_score(y_true=labels_val, y_score=predictions_scores)
    print("{:.2f}%\n".format(ROC_AUC*100))

    # Save the numpy array containing the coefficients as an HDF5 file
    # working_dir = os.path.dirname(os.path.abspath(__file__))
    # file_name = working_dir + "/output/CV_classifiers/clf_HDF5_files/norm_weights_{}_{}".format(title, data_size)
    # file_name = saving(file_name, format_type='h5', save=False)
    # h5f = h5py.File(file_name, 'w')
    # h5f.create_dataset('dataset_{}'.format(data_size), data=coef)


def get_arguments():
    """ Return the arguments given on the command line
    """

    # If you do not run from the command line
    # Read the data for both the deleterious and benign SNPs
    # Path directory for the benign SNPs always ending with combined_ben.h5
    # data_directory_ben = "/mnt/scratch/kersj001/data/output/1_mil_ben3/combined_ben.h5"
    # data_directory_ben = "/mnt/scratch/kersj001/data/output/10_thousand_ben/normalized.h5"
    # data_directory_ben = "/mnt/scratch/kersj001/data/output/test/test_ben2/combined_ben.h5"
    # data_directory_ben = "/mnt/scratch/kersj001/data/output/5/10_thousand_ben/normalized.h5"
    data_directory_ben_val = "/mnt/scratch/kersj001/data/output/ClinVar_ben/normalized.h5"
    data_directory_ben = "/mnt/scratch/kersj001/data/output/5/1_thousand_ben/normalized_new.h5"
    # data_directory_ben = "/mnt/scratch/kersj001/data/output/5/1_thousand_ben/combined_ben.h5"
    # data_directory_ben_val = "/mnt/scratch/kersj001/data/output/ClinVar_ben/combined_ben.h5"

    # Path directory for the deleterious SNPs always ending with combined_del.h5
    # data_directory_del = "/mnt/scratch/kersj001/data/output/1_mil_del/combined_del.h5"
    # data_directory_del = "/mnt/scratch/kersj001/data/output/10_thousand_del/normalized.h5"
    # data_directory_del = "/mnt/scratch/kersj001/data/output/test/test_del2/combined_del.h5"
    # data_directory_del = "/mnt/scratch/kersj001/data/output/5/10_thousand_del/normalized.h5"
    data_directory_del_val = "/mnt/scratch/kersj001/data/output/ClinVar_del/normalized.h5"
    data_directory_del = "/mnt/scratch/kersj001/data/output/5/1_thousand_del/normalized_new.h5"
    # data_directory_del = "/mnt/scratch/kersj001/data/output/5/1_thousand_del/combined_del.h5"
    # data_directory_del_val = "/mnt/scratch/kersj001/data/output/ClinVar_del/combined_del.h5"

    data_size = 2000

    # # Specify the options for running from the command line
    # parser = OptionParser()
    # # Specify the data directory for the benign and deleterious SNPs
    # parser.add_option("-b", "--ben", dest="benign", help="Path to the output of the 'combine_features.py' script that \
    #     generates a H5py file with the compressed numpy array containing feature scores of benign SNPs and its \
    #     neighbouring features", default="")
    # parser.add_option("-d", "--del", dest="deleterious", help="Path to the output of the 'combine_features.py' script \
    #     that generates a H5py file with the compressed numpy array containing feature scores of deleterious SNPs and \
    #     its neighbouring features", default="")
    # parser.add_option("-v", "--valben", dest="validation_benign", help="Path to the validation output of the \
    #         'combine_features.py' script that generates a H5py file with the compressed numpy array containing feature \
    #         scores of benign SNPs and its neighbouring features", default="")
    # parser.add_option("-e", "--valdel", dest="validation_deleterious", help="Path to the validation output of the \
    #         'combine_features.py' script that generates a H5py file with the compressed numpy array containing feature \
    #         scores of deleterious SNPs and its neighbouring features", default="")
    # parser.add_option("-n", "--datasize", dest="data_size", help="Integer that defines the number of samples in the \
    #     data set", default=2000)
    #
    # # Get the command line options for reading the data for both the benign and deleterious SNPs
    # (options, args) = parser.parse_args()
    # data_directory_ben = options.benign
    # data_directory_del = options.deleterious
    # data_directory_ben_val = options.validation_benign
    # data_directory_del_val = options.validation_deleterious
    # data_size = int(options.data_size)

    return data_directory_ben, data_directory_del, data_directory_ben_val, data_directory_del_val, data_size


if __name__ == "__main__":

    # Keep track of the running time
    start_time_script = time.time()

    # Get the given arguments
    data_directory_ben, data_directory_del, data_directory_ben_val, data_directory_del_val, data_size = get_arguments()

    # Read the HDF5 file back to a numpy array
    data_ben = data_reading(data_directory_ben)
    data_del = data_reading(data_directory_del)
    data_ben_val = data_reading(data_directory_ben_val)
    data_del_val = data_reading(data_directory_del_val)

    # Convert the NaN values into zero's
    data_ben = np.nan_to_num(data_ben)
    data_del = np.nan_to_num(data_del)
    data_ben_val = np.nan_to_num(data_ben_val)
    data_del_val = np.nan_to_num(data_del_val)

    # Parse the data for the logistic regression classification
    samples_SNPi, labels_SNPi, samples_neighbouring, labels_neighbouring, samples_SNPi_val, labels_SNPi_val, \
    samples_neighbouring_val, labels_neighbouring_val = data_parser(data_ben, data_del, data_ben_val, data_del_val)

    # Run the  logistic regression classifier for samples containing only the data of the SNP of interest and also for
    # the samples including neighbouring positional data
    SNP_or_neighbour = 'snp'
    print("NEIGHBOURS EXCLUDED")
    logistic_regression(samples_SNPi, labels_SNPi, samples_SNPi_val, labels_SNPi_val, data_size, SNP_or_neighbour)
    SNP_or_neighbour = 'neighbour'
    print("NEIGHBOURS INCLUDED")
    logistic_regression(samples_neighbouring, labels_neighbouring, samples_neighbouring_val, labels_neighbouring_val,
                        data_size, SNP_or_neighbour)

    # Get the complete running time of the script
    print("----- {} seconds -----".format(round(time.time() - start_time_script), 2))

    # Example for running from the command line
    """
    python PycharmProjects/thesis/data_checks/classifiers.py 
     --ben /mnt/scratch/kersj001/data/output/100_thousand_ben/combined_ben.h5 
     --del /mnt/scratch/kersj001/data/output/100_thousand_del/combined_del.h5  
     --datasize 200000
    """