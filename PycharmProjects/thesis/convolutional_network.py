#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date of creation: 05 April 2018
Date of last edit: 11 April 2018
Script for performing neural network containing a convolutional layer on the SNP data set

Output is a .txt file with statistic results of the neural network and a loss function plot
"""

import time
import h5py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm
from optparse import OptionParser


class NeuralNetwork(object):
    """ Neural network object to store the architecture and the trained weights
    """

    def __init__(self, all_samples, all_labels, fc_layers, do_layers, dropout_rate, act_func, learning_rate,
                 batch_size, feature_number, position_number, pooling_type):
        """ Initialize neural network object

        all_samples: numpy array, contains features scores with shape (number of samples, number of features) of the
         training samples
        all_labels: numpy array, contains class label 0 (benign SNPs) and 1 (deleterious SNPs) in the shape (number of
         samples, class dimension)
        fc_layers: list, contains integer values that defines the number of nodes for the fully connected layers
        do_layers: list, contains boolean values that defines if a fully connected layer is followed by a dropout layer
        dropout_rate: float, defines the dropout rate for the dropout layers
        act_func: Tensor, defines which activation function will be used for training the neural network
        learning_rate: float, defines the learning rate for training the network
        batch_size: integer, defines the batch size for training
        feature_number: integer, defines the number of features that is used for every nucleotide position
        position_number: integer, defines the considered nucleotide positions
        pooling_type: string, defines which type of pooling will be used after the convolutional layer
        """

        # Launch the TensorFlow session
        self.session = tf.Session()
        # self.session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4))

        # Get the number of considered nucleotide positions
        self.nt_positions = position_number

        # Get the number of used features for each nucleotide position
        self.nt_features = feature_number

        # Get the number of total features (number of considered nucleotides * number of features per nucleotide)
        self.n_total_features = all_samples.shape[-1]

        # Get the dimension of the class labels
        self.label_dimension = all_labels.shape[-1]

        # Define the neural network architecture
        self.create_graph(all_samples, all_labels, fc_layers, do_layers, dropout_rate, act_func, learning_rate,
                          batch_size, pooling_type)

        # Define the
        self.session.run(tf.global_variables_initializer())
        self.session.run(self.iter_initializer)

    def create_graph(self, all_samples, all_labels, nodes_per_layer, dropout_layers, dropout_rate, act_func,
                     learning_rate, batch_size, pooling_type):
        """ Defines the neural network architecture with name scoping for visualization in TensorBoard

        all_samples: numpy array, contains features scores with shape (number of samples, number of features) of the
         training samples
        all_labels: numpy array, contains class label 0 (benign SNPs) and 1 (deleterious SNPs) in the shape (number of
         samples, class dimension)
        nodes_per_layer: list, contains integer values that defines the number of nodes for the fully connected layers
        dropout_layers: list, contains boolean values that defines if a fully connected layer is followed by a dropout
         layer
        dropout_rate: float, defines the dropout rate for the dropout layers
        act_func: Tensor, defines which activation function will be used for training the neural network
        learning_rate: float, defines the learning rate for training the network
        batch_size: integer, defines the batch size for training
        pooling_type: string, defines which type of pooling will be used after the convolutional layer
        """

        with tf.variable_scope("Dataset"):
            # Get shuffled batch samples and labels
            request_batch_samples, request_batch_labels = self.get_batch(all_samples, all_labels, batch_size)

        with tf.variable_scope("Input"):
            # Create placeholders with default to use batch samples and labels
            batch_samples = tf.placeholder_with_default(request_batch_samples, [None, self.n_total_features],
                                                        name="BatchSamples")
            batch_labels = tf.placeholder_with_default(request_batch_labels, [None, self.label_dimension],
                                                       name="BatchLabels")

        with tf.variable_scope("ConvolutionalLayer"):
            # Reshape the data to match the convolutional layer format of [height x width x channel]
            # The tensor become a 4D of [batch size, height, width, channel]
            reshaped_batch_samples = tf.reshape(batch_samples, shape=[-1, self.nt_features, self.nt_positions, 1])

            # Keep track of the created layers
            layers = [reshaped_batch_samples]  # The batch samples will be used as input for the first hidden layer

            # Create the convolutional layer
            self.conv_layer = tf.layers.conv2d(inputs=layers[-1], filters=1, kernel_size=[1, self.nt_features],
                                               padding="same", activation=act_func, name="conv-layer")
            layers.append(self.conv_layer)

            # Create a pooling layer if it is defined
            if pooling_type == 'average pooling':
                self.pooling_layer = tf.reduce_mean(input_tensor=layers[-1], axis=1)
                layers.append(self.pooling_layer)
            elif pooling_type == 'max pooling':
                self.pooling_layer = tf.reduce_max(input_tensor=layers[-1], axis=1)
                layers.append(self.pooling_layer)
            elif pooling_type == 'no pooling':
                pass

            # Flatten the convolutional data in a 1D vector for the fully connected layer
            flatten_conv_layer = tf.contrib.layers.flatten(layers[-1])
            layers.append(flatten_conv_layer)

        with tf.variable_scope("HiddenLayers"):
            # The 'index_offset indicates which layer number it is; the offset increase if dropout layers are included
            index_offset = 1
            # Loop through the hidden layers nodes to define the hidden layers (including dropout layers)
            for index, n_units in enumerate(nodes_per_layer):
                # Create for every defined hidden node a fully connected layer
                name_layer = 'fc-layer{}'.format(index + index_offset)
                hidden_layer = tf.layers.dense(inputs=layers[-1], units=n_units, activation=act_func, name=name_layer)
                # Add the defined hidden layer to the list with all layers
                layers.append(hidden_layer)
                # Check if a fully connected layer is followed by a dropout layer
                if dropout_layers[index]:
                    index_offset += 1
                    name_layer = 'dr-layer{}'.format(index + index_offset)
                    hidden_layer = tf.layers.dropout(inputs=layers[-1], rate=dropout_rate, name=name_layer)
                    layers.append(hidden_layer)

        with tf.variable_scope("PredictionLayer"):
            # Define a prediction layer which has the class dimension as output
            pred = tf.layers.dense(layers[-1], self.label_dimension, activation=None)

        with tf.variable_scope("Cost"):
            # Sigmoid cross entropy can be used because the data allows binary classification
            self.cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=batch_labels)
            # Set the activaion function for prediction labels to evaluate the trained network
            self.prediction = tf.nn.sigmoid(pred)

        with tf.variable_scope("Optimizer"):
            # Define an optimize function to decrease the loss function
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # # Create a graph to visualize the architecture in TensorBoard
        # working_dir = os.path.dirname(os.path.abspath(__file__))
        # writer = tf.summary.FileWriter(working_dir + "/graphs/tensorboard", graph=self.session.graph)
        # writer.close()

    def get_batch(self, all_samples, all_labels, batch_size):
        """ Divide samples with corresponding labels in batches

        all_samples: numpy array, contains features scores with shape (number of samples, number of features) of the
         training samples
        all_labels: numpy array, contains class label 0 (benign SNPs) and 1 (deleterious SNPs) in the shape (number of
         samples, class dimension)
        batch_size: integer, defines the batch size for training

        :return:
        batch_samples: Tensor, of format (batch size, number of features)
        batch_labels: Tensor, of format (batch size, number of classes)
        """

        # Create a Tensor dataset object for the samples and labels
        samples_dataset = tf.data.Dataset.from_tensor_slices(all_samples)
        labels_dataset = tf.data.Dataset.from_tensor_slices(all_labels)

        # Combine the samples dataset with the labels dataset
        combined_dataset = tf.data.Dataset.zip((samples_dataset, labels_dataset))

        # Prevent that you run out of samples by repeating the dataset once
        combined_dataset = combined_dataset.repeat()

        # Shuffle the data
        combined_dataset = combined_dataset.shuffle(batch_size)

        # Create batches of your dataset
        combined_dataset = combined_dataset.batch(batch_size)

        # Initialize the dataset for TensorFlow
        iterator = combined_dataset.make_initializable_iterator()

        # Get the batch samples and labels operations
        batch_samples, batch_labels = iterator.get_next()

        # Convert the samples and labels to type float32 to use them in the convolutional layer
        batch_samples = tf.cast(batch_samples, tf.float32)
        batch_labels = tf.cast(batch_labels, tf.float32)

        # Make the iterator object global to initialize it from another function
        self.iter_initializer = iterator.initializer

        return batch_samples, batch_labels

    def train(self, n_iterations, test_samples, test_labels, training_samples, training_labels):
        """ Function that trains the neural network on the training data batches

        n_iterations: integer, defines the number of iteration for training the neural network
        test_samples: numpy array, contains features scores with shape (number of samples, number of features) of
         the test samples
        test_labels: numpy array, contains class label 0 (benign SNPs) and 1 (deleterious SNPs) of the test labels in
         the shape (number of samples, class dimension)
        training_samples: numpy array, contains features scores with shape (number of samples, number of features) of
         the training samples
        training_labels: numpy array, contains class label 0 (benign SNPs) and 1 (deleterious SNPs) of the training
         labels in the shape (number of samples, class dimension)
        """

        # Keep track of the running time for training the neural network
        start_time_network = time.time()

        # Train the neural network with the defined number of iterations on the training data batches
        all_training_loss = []
        all_test_loss = []
        for iteration in tqdm(range(n_iterations+1)):
            # The dataset create automatic batches, so there is no need to define the samples
            self.session.run(self.optimizer)
            # # Keep track of the training and test loss
            # training_loss = self.session.run(tf.reduce_mean(self.cost),
            #                                                 feed_dict={"Input/BatchSamples:0": training_samples,
            #                                                            "Input/BatchLabels:0": training_labels})
            # test_loss = self.session.run(tf.reduce_mean(self.cost), feed_dict={"Input/BatchSamples:0": test_samples,
            #                                                                    "Input/BatchLabels:0": test_labels})
            # # Store the loss in percentages
            # all_training_loss.append(training_loss*100)
            # all_test_loss.append(test_loss*100)

            # Check for every 100th iteration the loss
            if iteration % 1000 == 0:
                training_cost = self.session.run(tf.reduce_mean(self.cost))
                print("STEP {} | Training cost: {:.4f}".format(iteration, training_cost*100))
                test_accuracy = self.evaluate(evaluation_samples=test_samples, evaluation_labels=test_labels)
                print("\t\t   Test accuracy: {:.2f}%".format(test_accuracy[1]))

        trained_conv = self.session.run(self.conv_layer)
        print(trained_conv.shape)
        trained_pool = self.session.run(self.pooling_layer)
        print(trained_pool.shape)

        # Get the total running time of the neural network
        network_run_time = time.time() - start_time_network

        return network_run_time, all_training_loss, all_test_loss

    def evaluate(self, evaluation_samples, evaluation_labels):
        """ Function that evaluate the trained neural network with a test data set

        evaluation_samples: numpy array, contains features scores with shape (number of samples, number of features) of
         the test/validation samples
        evaluation_labels: numpy array, contains class label 0 (benign SNPs) and 1 (deleterious SNPs) in the shape
         (number of samples, class dimension)
        """

        # Use a feed dictionary to use the test samples/labels instead of the data from the Tensor dataset object
        prediction = self.session.run(self.prediction, feed_dict={"Input/BatchSamples:0": evaluation_samples,
                                                                  "Input/BatchLabels:0": evaluation_labels})

        # Get the ROC AUC score
        roc = roc_curve(y_true=evaluation_labels, y_score=prediction)
        area = auc(x=roc[0], y=roc[1])

        # Change the prediction scores into class labels to get the accuracy
        # Values that are higher than 0.5 get predicted class 1 and lower than 0.5 class label 0
        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] = 0
        # Get the evaluation accuracy
        accuracy = accuracy_score(y_true=evaluation_labels, y_pred=prediction)

        # Return the accuracies in percentages
        return area*100, accuracy*100

    def write_results(self, all_features, n_samples, n_neighbours, layers, act_func, dropout_rate, learning_rate,
                      iterations, optimum_idx, optimum_loss, start_test_cost, end_test_cost, accuracy_train,
                      accuracy_test, accuracy_val, ROC_AUC_train, ROC_AUC_test, ROC_AUC_val, run_time):
        """ Write the statistical results to the desired .txt file

        all_features: string, indicates if all conservation-based features are considered or only the PhastCon primate
         scores
        n_samples: integer, correspond to the number of samples in the data set
        n_neighbours: integer, indicates how many neighbouring positions are included
        layers: string, indicates which neural network architecture is used ('do'=dropout layer, 'fc ()'=fully connected
         layer with the number of nodes between the brackets
        act_func: string, indicates which activation function is used during the training
        dropout_rate: float, defines the dropout rate for the dropout layers
        learning_rate: float, strictness of learning while training the neural network
        iterations: integer, defines the number of iterations used for training the neural network
        optimum_idx: integer, defines the index value of the minimum loss
        optimum_loss: float, defines the minimum loss
        start_test_cost: float, defines the starting loss
        end_test_cost: float, defines the end loss of the trained network
        accuracy_train: float, training accuracy after fitting the logistic regression classifier
        accuracy_test: float, test accuracy after fitting the logistic regression classifier
        accuracy_val: float, validation accuracy after fitting the logistic regression classifier
        ROC_AUC_train: float, ROC AUC score of the training after fitting the logistic regression classifier
        ROC_AUC_test: float, ROC AUC score of the test after fitting the logistic regression classifier
        ROC_AUC_val: float, ROC AUC score of the validation samples after fitting the logistic regression classifier
        run_time: float, refers to the running time of the logistic regression classifier in seconds
        """

        # Get the file name where the statistical results will be written to
        working_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = working_dir + "/output_ANN/conv_results_ANN.txt"

        # Extend the file with the results in the corresponding data types
        with open(file_name, 'a') as output:
            output.write(
                "\n{:s}\t{:d}\t{:d}\t{:s}\t{:s}\t{:.2f}\t{:f}\t{:d}\t{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t"
                "{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.0f}".format(
                    all_features, n_samples, n_neighbours, layers, act_func, dropout_rate, learning_rate,
                    iterations, optimum_idx, optimum_loss, start_test_cost, end_test_cost, accuracy_train,
                    accuracy_test, accuracy_val, ROC_AUC_train, ROC_AUC_test, ROC_AUC_val, run_time))
        output.close()

    def loss_graph(self, training_costs, test_costs, learning_rate, training_accuracy, test_accuracy, val_accuracy,
                   layers, data_size, n_neighbours, dropout_layer, dropout_rate):
        """ Return plot of the decreasing error during the training of the neural network

        training_costs: list, contains training error costs
        test_costs: list, contains test error costs
        learning_rate: float, strictness of learning while training the neural network
        accuracy: float, defines the end training accuracy after training the neural network
        test_accuracy: float, defines the end test accuracy after training the neural network
        val_accuracy: float, defines the end validation accuracy after training the neural network
        layers: string, defines which layers are used with how many hidden nodes
        data_size: integer, defines the number of samples in the data
        n_neighbours: integer, indicates how many neighbours are included
        dropout_layer: list, contains boolean values that defines if a fully connected layer is followed by a dropout
         layer
        dropout_rate: float, defines the dropout rate for the dropout layers
        """

        plt.plot(training_costs, label="Training loss")
        plt.plot(test_costs, label="Test loss")
        plt.xlabel("Iterations", size='medium')
        plt.ylabel("Cost function (%)", size='medium')
        plt.suptitle("Cost function while training the neural network", size='medium', ha='center')
        if True in dropout_layer:
            plt.title("layers: {}, dropout rate: {}, learning rate: {}".format(layers, dropout_rate, learning_rate),
                      size='small', ha='center')
        else:
            plt.title("layers: {}, learning rate: {}".format(layers, learning_rate), size='small', ha='center')
        plt.figtext(0.77, 0.35, "Training accuracy\n{0:.2f}%".format(training_accuracy), size='medium')
        plt.figtext(0.77, 0.25, "Test accuracy\n{0:.2f}%".format(test_accuracy), size='medium')
        plt.figtext(0.77, 0.15, "Validation accuracy\n{0:.2f}%".format(val_accuracy), size='medium')
        if n_neighbours == 0:
            plt.figtext(0.77, 0.80, "Neighbours\nexcluded", size='medium')
        else:
            plt.figtext(0.77, 0.80, "{} neighbours\nincluded".format(n_neighbours), size='medium')
        plt.figtext(0.77, 0.70, "{}\nsamples".format(data_size))
        plt.legend(loc='right', bbox_to_anchor=(1.39, 0.5))
        plt.subplots_adjust(right=0.75)
        working_dir = os.path.dirname(os.path.abspath(__file__))
        saving(working_dir + "/output_ANN/loss_plots/{}_error_{}".format(n_neighbours, data_size))


def saving(file_path, file_type='png'):
    """ Saves the plot in the correct directory with an unique name

    file_path: string, path direction to where the plot will be saved
    file_type: string, desired format of the file, default='png'
    """

    index_saving = 1
    while os.path.exists(file_path + "_{}.{}".format(index_saving, file_type)):
        index_saving += 1

    plt.savefig(file_path + "_{}.{}".format(index_saving, file_type))


def data_reading(data_file):
    """ Reads the .H5py file containing the data back as a nunmpy array

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
        index_SNPi = (data.shape[2] - 1) / 2  # -1 for the SNP of interest
        samples = data[:, :, int(index_SNPi)]
        # Define the number of considered nucleotide positions
        n_positions = 1

    # Get the features of the SNP of interest and neighbouring positions
    else:
        # The data should fit in a 2D array for performing neural network. The number of samples should be stay, and
        # the number of features will be the number of features times the number of nucleotides
        samples = data.reshape([data.shape[0], -1])
        # Define the number of considered nucleotide positions
        n_positions = data.shape[2]

    # Get the number of used features
    n_features = data.shape[1]

    return samples, n_features, n_positions


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

    # Create float numbers for the labels
    labels = labels.astype(float)

    # Convert the data into a numpy array
    # One hot encoded vector is not necessary, because the data is binary
    labels = np.reshape(labels, [-1, 1])

    return labels


def define_parameters(all_features, act_func, dropout, fc_layer_units, training_samples, batch_perc, pooling):
    """ Convert some parameters in the right format

    act_funct: string, defines which activation function will  be used for training the neural network
    dropout: list, contains boolean values that defines if a fully connected layer is followed by a dropout layer
    fc_layers_units: list, contains integer values that defines the number of nodes for the fully connected layers
    training_samples: numpy array, contains features scores with shape (samples, number of features * number of
     neighbouring positions) of the training samples
    batch_perc: flaot, defines the percentage of the training samples that will be used as batch size
    pooling: string, refers to the type of pooling layer
    """

    # Get the variables in the right format
    all_features = all_features.lower()
    if all_features == 'y' or all_features == 'yes':
        all_features = "all conservation-based scores"
    else:
        all_features = "only PhastCon primate scores"

    # Get the batch size
    batch_size = int(training_samples.shape[0] * batch_perc)

    # Get the layer nodes in correct format
    int_layer_units = []
    units = fc_layer_units.split(',')
    for unit in units:
        int_layer_units.append(int(unit))

    # Get the dropout layers in correct format
    dropout_booleans = []
    dropouts = dropout.split(',')
    for layer in dropouts:
        layer = layer.lower()
        if layer == 'f' or layer == 'false':
            dropout_booleans.append(False)
        else:
            dropout_booleans.append(True)

    # Get the layer names of the neural network architecture
    layers = ['conv']
    for index, nodes in enumerate(int_layer_units):
        layers.append('fc ({})'.format(nodes))
        if dropout_booleans[index]:
            layers.append('do')
    layers = ' - '.join(layers)

    # Get the right activation function
    act_func = act_func.lower()
    if act_func == 'sigmoid' or act_func == 'sig' or act_func == 's':
        act_func = tf.nn.sigmoid
        act_title = 'sigmoid'
    elif act_func == 'relu' or act_func == 'r':
        act_func = tf.nn.relu
        act_title = 'ReLU'
    elif act_func == 'leakyrelu' or act_func == 'leaky' or act_func == 'l':
        act_func = tf.nn.leaky_relu
        act_title = 'Leaky ReLU'
    elif act_func == 'tanh' or act_func == 'tan' or act_func == 't':
        act_func = tf.tanh
        act_title = 'tanH'
    else:
        act_func = None
        act_title = 'linear'

    pooling = pooling.lower()
    if pooling == 'average' or pooling == 'avg' or pooling == 'a' or pooling == 'mean':
        pooling_name = 'average pooling'
    elif pooling == 'max' or pooling == 'm':
        pooling_name = 'max pooling'
    else:
        pooling_name = 'no pooling'

    return all_features, act_func, act_title, batch_size, layers, dropout_booleans, int_layer_units, pooling_name


def get_arguments():
    """ Return the arguments given on the command line
    """

    # Specify the variable if you do not run from the command line
    data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/5_200000.h5"
    val_data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/5_26172.h5"
    all = 'yes'
    snp_neighbour = 'neighbour'
    # snp_neighbour = 'snp'
    learning_rate = 0.01
    dropout_rate = 0.5
    activation_function = 't'
    batch_percentage = 0.01
    iterations = 10000
    fc_nodes = '10'
    dropout_layer = 't'
    pooling_layer = 'a'

    # # Specify the options for running from the command line
    # parser = OptionParser()
    # # Specify the data directory for the benign and deleterious SNPs
    # parser.add_option("-d", "--data", dest="data", help="Path to the output of the normalized feature scores of \
    #     deleterious SNPs and its neighbouring features", default="")
    # # Specify the data directory for the validation samples
    # parser.add_option("-v", "--valdata", dest="validation_data", help="Path to the normalized validation samples",
    #     default="")
    # # Specify if only the PhastCon primate scores are considered or all the conservation-based features
    # parser.add_option("-f", "--all", dest="all_features", help="String (yes or no) that indicates if all conservation-\
    #     based features are considered", default="yes")
    # # Specify if the neural network will included neighbouring positions
    # parser.add_option("-s", "--snp", dest="snp_neighbour", help="String that indicates if the surrounding neighbours \
    #     will be included ('n') or excluded ('s')", default="n")
    # # Specify the learning rate
    # parser.add_option("-l", "--learning", dest="learning_rate", help="Float that defines the learning rate of for \
    #     training the neural network", default=0.01)
    # # Specify the dropout rate
    # parser.add_option("-r", "--droprate", dest="dropout_rate", help="Float that defines the dropout rate for the \
    #     dropout layers", default=0.5)
    # # Specify the activation function
    # parser.add_option("-a", "--act", dest="activation", help="String that refers to the activation function that will \
    #     be used for training the neural network", default="relu")
    # # Specifiy the number of iterations
    # parser.add_option("-i", "--iter", dest="iterations", help="Integer that defines the number of iterations for \
    #     training the neural network", default=1000)
    # # Specify the percentage of training data that will be use as batch size
    # parser.add_option("-b", "--batch", dest="batch_perc", help="Float that defines the percentage of training samples \
    #     that will be used as batch size", default=0.01)
    # # Specify the nodes for the fully connected layers
    # parser.add_option("-n", "--nodes", dest="fc_nodes", help="List that contains integer values that defines the \
    #     number of nodes for the fully connected layer", default="4")
    # # Specify if dropout layers occurs after a fully connected layer
    # parser.add_option("-o", "--dropout", dest="dropout_layers", help="List that contains boolean values that defines \
    #     if a fully connected layer is followed by a dropout layer", default="False")
    # parser.add_option("-p", "--pooling", dest="pooling_layer", help="String that refers if and what type of pooling \
    #     layer is used", default="none")
    #
    # # Get the command line options for reading the data for both the benign and deleterious SNPs
    # (options, args) = parser.parse_args()
    # data_directory = options.data
    # val_data_directory = options.validation_data
    # all = options.all_features
    # snp_neighbour = options.snp_neighbour
    # learning_rate = float(options.learning_rate)
    # dropout_rate = float(options.dropout_rate)
    # activation_function = options.activation
    # batch_percentage = float(options.batch_perc)
    # iterations = int(options.iterations)
    # fc_nodes = options.fc_nodes
    # dropout_layer = options.dropout_layers
    # pooling_layer = options.pooling_layer

    return data_directory, val_data_directory, all, snp_neighbour, learning_rate, dropout_rate, activation_function, \
           batch_percentage, iterations, fc_nodes, dropout_layer, pooling_layer


if __name__ == "__main__":
    # Keep track of the running time
    start_time = time.time()

    # Get the given arguments including the hyperparameters for the neural network
    data_directory, val_data_directory, all_features, snp_neighbour, learning_rate, dropout_rate, \
        activation_function_name, batch_percentage, iterations, fc_nodes, dropout_layers, pooling_layer_type = \
        get_arguments()

    # Read the HDF5 file back to a numpy array
    data, data_size, n_neighbours = data_reading(data_file=data_directory)
    val_data, _, _ = data_reading(data_file=val_data_directory)

    # Get the number of considered neighbouring positions
    snp_neighbour = snp_neighbour.lower()
    if snp_neighbour == "s" or snp_neighbour == "snp":
        n_neighbours = 0
    else:
        n_neighbours = n_neighbours

    # Parse the data into samples which either consider neighbouring positions or not
    samples, n_features, n_positions = data_parser(data=data, snp_neighbour=n_neighbours)
    samples_val, _, _ = data_parser(data=val_data, snp_neighbour=n_neighbours)

    # Get the data labels
    labels = data_labels(data=data)
    labels_val = data_labels(data=val_data)

    # Define a training and test set
    test_size = 0.1  # training is set on 90%
    training_data, test_data, training_labels, test_labels = train_test_split(samples, labels, test_size=test_size)

    # Get the parameters for the neural network in correct format
    feature_title, act_func, act_title, batch_size, layer_names, dropout_booleans, fc_layer_units, pooling_layer_name \
        = define_parameters(all_features=all_features, act_func=activation_function_name, dropout=dropout_layers,
                            fc_layer_units=fc_nodes, training_samples=training_data, batch_perc=batch_percentage,
                            pooling=pooling_layer_type)

    # Create the neural network
    nn = NeuralNetwork(all_samples=training_data, all_labels=training_labels, fc_layers=fc_layer_units,
                       do_layers=dropout_booleans, dropout_rate=dropout_rate, act_func=act_func,
                       learning_rate=learning_rate, batch_size=batch_size, feature_number=n_features,
                       position_number=n_positions, pooling_type=pooling_layer_name)

    # Train the neural network
    run_time, training_loss, test_loss = nn.train(n_iterations=iterations, test_samples=test_data,
                                                  test_labels=test_labels, training_samples=training_data,
                                                  training_labels=training_labels)

    # Evaluate the neural network with the test data and validation data
    training_roc, training_accuracy, = nn.evaluate(evaluation_samples=training_data, evaluation_labels=training_labels)
    test_roc, test_accuracy, = nn.evaluate(evaluation_samples=test_data, evaluation_labels=test_labels)
    val_roc, val_accuracy = nn.evaluate(evaluation_samples=samples_val, evaluation_labels=labels_val)
    print("\nAccuracy training data: {:.2f}".format(training_accuracy))
    print("Accuracy test data: {:.2f}".format(test_accuracy))
    print("Accuracy validation data: {:.2f}".format(val_accuracy))
    print("\nAUC training data: {:.2f}".format(training_roc))
    print("AUC test data: {:.2f}".format(test_roc))
    print("AUC validation data: {:.2f}".format(val_roc))

    # Get some statistics about the cost function
    try:
        start_test_cost = test_loss[0]
        end_test_cost = test_loss[-1]
        min_test_cost = min(test_loss)
        min_test_cost_idx = test_loss.index(min_test_cost)
        print("\noptimum number of iterations: {}".format(min_test_cost_idx))
    except:
        start_test_cost = 0
        end_test_cost = 0
        min_test_cost = 0
        min_test_cost_idx = 0

    # Write the statistical outcomes to a defined file
    nn.write_results(all_features=feature_title, n_samples=data_size, n_neighbours=n_neighbours, layers=layer_names,
                     act_func=act_title, dropout_rate=dropout_rate, learning_rate=learning_rate, iterations=iterations,
                     optimum_idx=min_test_cost_idx, optimum_loss=min_test_cost, start_test_cost=start_test_cost,
                     end_test_cost=end_test_cost, accuracy_train=training_accuracy, accuracy_test=test_accuracy,
                     accuracy_val=val_accuracy, ROC_AUC_train=training_roc, ROC_AUC_test=test_roc, ROC_AUC_val=val_roc,
                     run_time=run_time)

    # # Create a plot of the loss function
    # nn.loss_graph(training_costs=training_loss, test_costs=test_loss, learning_rate=learning_rate, training_accuracy=
    #               training_accuracy, test_accuracy=test_accuracy, val_accuracy=val_accuracy, layers=layer_names,
    #               data_size=data_size, n_neighbours=n_neighbours, dropout_layer=dropout_booleans, dropout_rate=
    #               dropout_rate)
