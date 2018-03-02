#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date: 2 March 2018
Script for performing Support Vector Machine, Logistic Regression, Decision Tree, and Random Forest classification

Output is .txt file containing the test accuracy with the standard deviation and the validation accuracy for a \
specified classifier
"""

import time
import os
import h5py
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from optparse import OptionParser


def tests(clf_name, data_ben, data_del, data_ben_val, data_del_val, data_size, SNP_or_neighbour='n',
          kernel_size="N.A."):
    """ Function reshaped the data in order to perform classification with and without neighbouring positions

    clf_name: string, defines which classifier should be run
    data_ben: numpy array with shape (number of samples, number of features, number of nucleotides), the samples are\
     benign SNPs
    data_del: numpy array with shape (number of samples, number of features, number of nucleotides), the samples are\
     deleterious SNPs
    data_ben_val: numpy array with shape (number of samples, number of features, number of nucleotides), the samples \
     are validation benign samples obtained from the Exon Sequencing Project
    data_del_val: numpy array with shape (number of samples, number of features, number of nucleotides), the samples \
     are validation deleterious samples obtained from the ClinVar database
    data_size: integer, defines the number of samples in the data
    SNP_or_neighbour: string, indicates if the classifier will run with only the features of the SNP of interest or \
     also includes the features of the neighbouring positions
    kernel_size: float or string, indicated the kernel size for the support vector machine classifier with radial \
     basis function (other classifiers do not need a specific kernel size), default="N.A."
    """

    # Reshape the data into a 2D numpy array containing either only the samples of the SNP of interest or samples
    # including neighbouring positions as well
    SNPi_ben, SNPi_del, data_ben_2D, data_del_2D = data_reshape(data_ben, data_del)
    SNPi_ben_val, SNPi_del_val, data_ben_2D_val, data_del_2D_val = data_reshape(data_ben_val, data_del_val)

    # Get the samples and labels for both the training data and validation data
    samples_SNPi, labels_SNPi = samples_labels(SNPi_ben, SNPi_del)
    samples_neighbouring, labels_neigbouring = samples_labels(data_ben_2D, data_del_2D)
    samples_SNPi_val, labels_SNPi_val = samples_labels(SNPi_ben_val, SNPi_del_val)
    samples_neighbouring_val, labels_neigbouring_val = samples_labels(data_ben_2D_val, data_del_2D_val)

    # Run the specified classifier
    SNP_or_neighbour = SNP_or_neighbour.lower()
    # Run the classifier for samples containing only the data of the SNP of interest and also for the samples including
    # neighbouring positional data
    if SNP_or_neighbour == 'ns' or SNP_or_neighbour == 'sn':
        SNP_or_neighbour = 's'
        classifier(clf_name, samples_SNPi, labels_SNPi, samples_SNPi_val, labels_SNPi_val, data_size, SNP_or_neighbour,
                   kernel_size)
        SNP_or_neighbour = 'n'
        classifier(clf_name, samples_neighbouring, labels_neigbouring, samples_neighbouring_val, labels_neigbouring_val,
                   data_size, SNP_or_neighbour, kernel_size)
    # Run the classifier only for samples containing feature scores of the SNP of interest
    elif SNP_or_neighbour == 's' or SNP_or_neighbour == 'snp':
        classifier(clf_name, samples_SNPi, labels_SNPi, samples_SNPi_val, labels_SNPi_val, data_size, SNP_or_neighbour,
                   kernel_size)
    # Run the classifier only for samples containing feature scores of the SNP of interest and neighbouring positions
    else:
        classifier(clf_name, samples_neighbouring, labels_neigbouring, samples_neighbouring_val, labels_neigbouring_val,
                   data_size, SNP_or_neighbour, kernel_size)


def data_reshape(data_ben, data_del):
    """ Return a reshaped numpy array containing only the features of the SNP of interest, or includes the neighbouring
     positions

    data_ben: numpy array with shape (number of samples, number of features, number of nucleotides), the samples are\
     benign SNPs
    data_del: numpy array with shape (number of samples, number of features, number of nucleotides), the samples are\
     deleterious SNPs
    """

    # Get only the features of the SNP of interest, which is located at the middle position of the data
    index_SNPi_ben = (data_ben.shape[2] - 1) / 2  # -1 for the SNP of interest
    index_SNPi_del = (data_del.shape[2] - 1) / 2
    SNPi_ben = data_ben[:, :, int(index_SNPi_ben)]
    SNPi_del = data_del[:, :, int(index_SNPi_del)]

    # Reshape the data sets including neighbouring SNPs to a 2D array
    # The original number of samples should be the same as previous, and the number of features will be the number of
    # features times the number of nucleotides
    data_ben_2D = data_ben.reshape([data_ben.shape[0], -1])
    data_del_2D = data_del.reshape([data_del.shape[0], -1])

    return SNPi_ben, SNPi_del, data_ben_2D, data_del_2D


def samples_labels(data_ben, data_del):
    """ Return the benign and deleterious SNP samples into one numpy array and the corresponding labels in a numpy array

    data_ben: numpy array with shape (number of samples, number of features, number of nucleotides), the samples are\
     benign SNPs
    data_del: numpy array with shape (number of samples, number of features, number of nucleotides), the samples are\
     deleterious SNPs
    """

    # Combine the benign and deleterious SNPs to 1 array
    samples = np.concatenate((data_ben, data_del), axis=0)  # 0 to put the arrays behind each other

    # Get the corresponding labels; the SNPs that are benign have class label 0, and the deleterious SNPs class label 1
    # The number of samples is located at the first position of the data shape
    labels_ben = [0] * data_ben.shape[0]
    label_del = [1] * data_del.shape[0]
    labels = np.array(labels_ben + label_del)

    return samples, labels


def classifier(clf_name, samples, labels, samples_val, labels_val, n_samples, title, kernel_size):
    """ Function for fitting the classifier and getting the test and validation accuracy written to a .txt file

    clf_name: string, defines which classifier should be run
    samples: numpy array, contains features scores with shape (samples, number of features * number of neighbouring \
     positions)
    labels: numpy array, contains data labels corresponding to the samples
    samples_val: numpy array, contains features scores with shape (validation samples, number of features * number of \
     neighbouring positions)
    labels_val: numpy array, contains data labels corresponding to the validation samples
    n_samples: integer, defines the number of samples in the data
    title: string, indicates if the classifier will run with only the features of the SNP of interest or also includes \
     the features of the neighbouring positions
    kernel_size: float or string, indicated the kernel size for the support vector machine classifier with radial \
     basis function (other classifiers do not need a specific kernel size), default="N.A."
    """

    # Take track of how long the classifier runs
    start_time = time.time()

    # Get the correct classifier
    clf, clf_name, save_name = find_classifier_name(clf_name, kernel_size)

    # Perform cross validation on the classifier
    clf, all_scores = cross_validation(clf, samples, labels)

    # Get the test accuracy and standard deviation
    accuracy = all_scores.mean()
    std = all_scores.std() * 2

    # Validate the trained classifier with the validation set
    predictions_val = clf.predict(X=samples_val)
    accuracy_val = accuracy_score(y_true=labels_val, y_pred=predictions_val)
    # Random forest and decision tree have another function for calculation the ROC-AUC
    if clf_name == "Random forest" or clf_name == "Decision tree":
        predictions_scores = clf.predict_proba(X=samples_val)
        # Get only the prediction of one class to perform the 'roc_auc_score' function
        predictions_class1 = predictions_scores[:, 1]
        ROC_AUC = roc_auc_score(y_true=labels_val, y_score=predictions_class1)
    else:
        predictions_scores = clf.decision_function(X=samples_val)
        ROC_AUC = roc_auc_score(y_true=labels_val, y_score=predictions_scores)

    # Write a text file in the right directory for saving the result of the classifier
    working_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = working_dir + "/output/CV_classifiers/{}/accuracy_{}_val+ROC".format(save_name, data_size)
    file_name = saving(file_name)
    writing(clf_name, file_name, accuracy, std, kernel_size, n_samples, title, accuracy_val, ROC_AUC)

    print(clf_name, title, kernel_size, "\ttime: {} seconds".format(round(time.time() - start_time), 2))


def find_classifier_name(clf_name, kernel_size):
    """ Return the classifier model and the name of classifier that will be used in the rest of the script

    clf_name: string, defines which classifier should be run
    kernel_size: float or string, indicated the kernel size for the support vector machine classifier with radial \
     basis function (other classifiers do not need a specific kernel size), default="N.A."
    """

    # Make sure the classifier name is comparable
    clf_name = clf_name.lower()
    # Define the correct classifier
    # For Logistic Regression
    if 'log' in clf_name:
        clf = linear_model.LogisticRegression()
        clf_name = "Logistic regression"
        save_name = "logistic"
    # For Decision Tree
    elif 'tree' in clf_name:
        clf = tree.DecisionTreeClassifier(max_depth=5)
        clf_name = "Decision tree"
        save_name = "decision_tree"
    # For Random Forest
    elif 'forest' in clf_name or clf_name == 'rf':
        clf = RandomForestClassifier(max_depth=5)
        clf_name = "Random forest"
        save_name = "random_forest"
    # For Support Vector Machine with Radial Basis Function (RBF)
    elif 'rbf' in clf_name:
        clf = svm.SVC(kernel='rbf', gamma=kernel_size)
        clf_name = "SVM rbf"
        save_name = "SVM_rbf"
    # For Support Vector Machine with Linear function
    else:
        clf = svm.SVC(kernel='linear')
        clf_name = "SVM linear"
        save_name = "SVM_linear"

    return clf, clf_name, save_name


def cross_validation(clf, samples, labels):
    """ Fit the classifier model with 10-Fold cross validation and gives the test accuracy back

    clf: sklearn class, defines the classifier model
    samples: numpy array, contains features scores with shape (samples, number of features * number of neighbouring \
     positions)
    labels: numpy array, contains data labels corresponding to the samples
    """

    # Defines the K-fold
    k_fold = KFold(n_splits=10, shuffle=True)

    # Keep track of the test accuracies
    all_scores = []
    for train_index, test_index in k_fold.split(samples):
        # Get a training and test set for the samples and corresponding labels
        x_train, x_test = samples[train_index], samples[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # Fit the classifier and give the test accuracy
        score = clf.fit(X=x_train, y=y_train).score(X=x_test, y=y_test)
        all_scores.append(score)

    # Convert the test accuracy list into a numpy array
    all_scores = np.array(all_scores)

    return clf, all_scores


def saving(file_path, format_type="txt"):
    """ Create a .txt file with header line for saving the output values of the classifier

    file_path: string, directory where the .txt file can be written to
    format_type: string, type of the created file, default="txt"
    """

    # Get an unique file name
    index_saving = 1
    while os.path.exists(file_path+"_{}.{}".format(index_saving, format_type)):
        index_saving += 1

    # Write the header line for the created file name
    file_name = os.path.join(file_path+"_{}.{}".format(index_saving, format_type))
    with open(file_name, 'w') as file:
        file.write("classifier\tincluding or excluding neighbours\tdata size\tkernel size\taccuracy\t"
                    "standard deviation\taccuracy validation\tROC-AUC")
        file.close()

    return file_name


def writing(clf_name, file_name, accuracy, std, kernel_size, n_samples, title, accuracy_val, ROC_AUC):
    """ Write the output values in correct format to the specified .txt file

    clf_name: string, defines which classifier was used
    file_name: string, directory ending with the file name where the output should be written to
    accuracy: float, defines the test accuracy after fitting the classifier model
    std: float, defines the standard deviation after perform cross validation
    kernel_size: float or string, indicated the kernel size for the support vector machine classifier with radial \
     basis function (other classifiers do not need a specific kernel size), default="N.A."
    n_samples: integer, defines the number of samples in the data
    title: string, indicates if the classifier was run with only the features of the SNP of interest or also includes \
     the features of the neighbouring positions
    accuracy_val: float, defines the validation accuracy after training the classifier model
    ROC_AUC: float, defines the validation ROC-AUC score after training the classifier model
    """

    # Get the correct title name
    if title == "s" or title == 'snp':
        title = "excluding"
    else:
        title = "including"

    # ROC-AUC is in string format when it is not defined; and a float when it is specified
    # Convert the ROC-AUC to string format for writing it to the .txt file
    if type(ROC_AUC) != str:
        ROC_AUC = str("{:.2f}".format(ROC_AUC))

    # Write the outcomes to the correct .txt file with the corresponding data types
    with open(file_name, 'a') as output:
        output.write("\n{:s}\t{:s}\t{:d}\t{:s}\t{:.2f}\t{:.4f}\t{:.2f}\t{:s}".format(clf_name, title, n_samples,
                     str(kernel_size), accuracy, std, accuracy_val, ROC_AUC))
    output.close()

    return title


def data_reading(data_directory):
    """ Reads the .H5py file containing the data back as a nunmpy array

    data_directory: string, working directory where the data can be found
    """

    # Read the data
    h5f = h5py.File(data_directory, 'r')

    # The data set name is the name of the path where the data file can be found
    data = h5f[data_directory.split('/')[-2]][:]

    # Close the H5py file
    h5f.close()

    return data


def get_arguments():
    """ Return the arguments given on the command line
    """

    # If you do not run from the command line
    """
    # Read the data for both the deleterious and benign SNPs
    # Path directory for the benign SNPs always ending with combined_ben.h5
    # data_directory_ben = "/mnt/scratch/kersj001/data/output/1_mil_ben3/combined_ben.h5"
    # data_directory_ben = "/mnt/scratch/kersj001/data/output/10_thousand_ben/combined_ben.h5"
    # data_directory_ben = "/mnt/scratch/kersj001/data/output/1_thousand_ben/combined_ben.h5"
    data_directory_ben = "/mnt/scratch/kersj001/data/output/test/test_ben2/combined_ben.h5"
    data_directory_ben_val = "/mnt/scratch/kersj001/data/output/ClinVar_ben/combined_ben.h5"
    

    # Path directory for the deleterious SNPs always ending with combined_del.h5
    # data_directory_del = "/mnt/scratch/kersj001/data/output/1_mil_del/combined_del.h5"
    # data_directory_del = "/mnt/scratch/kersj001/data/output/10_thousand_del/combined_del.h5"
    # data_directory_del = "/mnt/scratch/kersj001/data/output/1_thousand_del/combined_del.h5"
    data_directory_del = "/mnt/scratch/kersj001/data/output/test/test_del2/combined_del.h5"
    data_directory_del_val = "/mnt/scratch/kersj001/data/output/ClinVar_del/combined_del.h5"

    data_size = 20
    snp_neighbour = 'ns'
    clf_name = 'log'
    kernel_size = ""
    try:
        kernel_size = float(kernel_size)
    except:
        kernel_size = "N.A."
    """

    # Specify the options for running from the command line
    parser = OptionParser()
    # Specify the data directory for the benign and deleterious SNPs
    parser.add_option("-b", "--ben", dest="benign", help="Path to the output of the 'combine_features.py' script that \
        generates a H5py file with the compressed numpy array containing feature scores of benign SNPs and its \
        neighbouring features", default="")
    parser.add_option("-d", "--del", dest="deleterious", help="Path to the output of the 'combine_features.py' script \
        that generates a H5py file with the compressed numpy array containing feature scores of deleterious SNPs and \
        its neighbouring features", default="")
    parser.add_option("-v", "--valben", dest="validation_benign", help="Path to the validation output of the \
        'combine_features.py' script that generates a H5py file with the compressed numpy array containing feature \
        scores of benign SNPs and its neighbouring features", default="")
    parser.add_option("-e", "--valdel", dest="validation_deleterious", help="Path to the validation output of the \
        'combine_features.py' script that generates a H5py file with the compressed numpy array containing feature \
        scores of deleterious SNPs and its neighbouring features", default="")
    parser.add_option("-n", "--datasize", dest="data_size", help="Integer that defines the number of samples in the \
        data set", default=2000)
    parser.add_option("-c", "--classifier", dest="clf_name", help="Type of classifier. Can be random forest (RF), \
        decision tree (tree), logistic regression (log), SVM with radial basis function (rbf) or SVM with linear \
        function (lin)", default="")
    parser.add_option("-s", "--snp", dest="snp_neighbour", help="String that indicates if the surrounding neighbours \
        will be included ('n') or excluded ('s')", default="n")
    parser.add_option("-k", "--kernelsize", dest="kernel_size", help="Float that defines the kernel size of the rdf \
            svm classifier", default=0.01)

    # Get the command line options for reading the data for both the benign and deleterious SNPs
    (options, args) = parser.parse_args()
    data_directory_ben = options.benign
    data_directory_del = options.deleterious
    data_directory_ben_val = options.validation_benign
    data_directory_del_val = options.validation_deleterious
    data_size = int(options.data_size)
    clf_name = options.clf_name
    snp_neighbour = options.snp_neighbour
    # Get the correct (type of the) kernel size
    try:
        kernel_size = float(options.kernel_size)
    except:
        kernel_size = "N.A."

    return data_directory_ben, data_directory_del, data_directory_ben_val, data_directory_del_val, data_size, \
           clf_name, snp_neighbour, kernel_size


if __name__ == "__main__":

    # Keep track of the running time
    start_time_script = time.time()

    # Get the given arguments
    data_directory_ben, data_directory_del, data_directory_ben_val, data_directory_del_val, data_size, clf_name, \
    snp_neighbour, kernel_size = get_arguments()

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

    # Run the classifier tests
    tests(clf_name, data_ben, data_del, data_ben_val, data_del_val, data_size, SNP_or_neighbour=snp_neighbour,
          kernel_size=kernel_size)

    # Get the complete running time of the script
    print("----- {} seconds -----".format(round(time.time() - start_time_script), 2))

    # Example for running from the command line
    """
    python PycharmProjects/thesis/data_checks/classifiers.py 
     --ben /mnt/scratch/kersj001/data/output/100_thousand_ben/combined_ben.h5 
     --del /mnt/scratch/kersj001/data/output/100_thousand_del/combined_del.h5 
     --valben /mnt/scratch/kersj001/data/output/ClinVar_ben/combined_ben.h5 
     --valdel /mnt/scratch/kersj001/data/output/ClinVar_del/combined_del.h5 
     --datasize 200000 --classifier rbf --snp n --kernelsize 0.0001
    """