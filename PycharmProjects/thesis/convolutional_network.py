#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date of creation: 05 April 2018
Date of last edit: 14 May 2018
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

    def __init__(self, all_samples, all_labels, fc_layers, do_layers, dropout_rate, act_func, learning_rate, batch_size,
                 feature_number, position_number, pooling_type, filter_number, filter_slide, filter_direction,
                 padding_type):
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
        filter_number: list, contains integer values that defines the number of filter that will be used for each
         convolutional layer
        filter_slide: list, contains integer values that defines the filter width that will be used for each
         convolutional layer
        filter_direction: string, defines in which direction the filter is sliding over the convolutional layer
        padding_type: string, defines which type of padding is used
        """

        # Launch the TensorFlow session
        # self.session = tf.Session()
        self.session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=6))

        # Get the number of considered nucleotide positions
        self.nt_positions = position_number

        # Get the number of used features for each nucleotide position
        self.nt_features = feature_number

        # Get the number of total features (number of considered nucleotides * number of features per nucleotide)
        self.n_total_features = all_samples.shape[-1]

        # Get the dimension of the class labels
        self.label_dimension = all_labels.shape[-1]

        # Get the number of filters that are used for each convolutional layer
        self.filter_number = filter_number

        # Get the filter width for each convolutional layer
        self.filter_slide = filter_slide

        # Get the sliding direction of the filter
        self.filter_direction = filter_direction

        # Define the loss, to perform early stopping of the neural network
        self.best_loss = 100
        self.stopping_step = 0

        # Define the neural network architecture
        self.create_graph(all_samples, all_labels, fc_layers, do_layers, dropout_rate, act_func, learning_rate,
                          batch_size, pooling_type, padding_type)

        # Define the
        self.session.run(tf.global_variables_initializer())
        self.session.run(self.iter_initializer)

    def create_graph(self, all_samples, all_labels, nodes_per_layer, dropout_layers, dropout_rate, act_func,
                     learning_rate, batch_size, pooling_type, padding_type):
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
        padding_type: string, defines which type of padding is used
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
            self.all_input_samples = tf.placeholder(tf.float32, shape=[None, self.n_total_features],
                                                    name="AllDataInput")
            self.all_input_labels = tf.placeholder(tf.float32, shape=[None, self.label_dimension], name="AllLabelInput")

        with tf.variable_scope("ConvolutionalLayer"):
            # Reshape the data to match the convolutional layer format of [height x width x channel]
            # The tensor become a 4D of [batch size, height, width, channel]
            reshaped_batch_samples = tf.reshape(batch_samples, shape=[-1, self.nt_features, self.nt_positions, 1])
            # Keep track of the created layers
            layers = [reshaped_batch_samples]  # The batch samples will be used as input for the first hidden layer
            if self.filter_direction == 'both':
                # Create the first layer with using a filter sliding over positions
                weights1 = tf.Variable(tf.random_normal([self.nt_features, self.filter_slide[0], 1,
                                                        self.filter_number[0]]), name='filter-weights0')
                tf.add_to_collection('conv_weights', weights1)
                conv_layer1 = tf.nn.conv2d(input=layers[0], filter=weights1, strides=[1, 1, 1, 1], padding=padding_type,
                                           name="conv-layer0")
                layers.append(conv_layer1)
                # Create a second layer with using a filter sliding over features
                weights2 = tf.Variable(tf.random_normal([self.filter_slide[1], self.nt_positions, 1,
                                                         self.filter_number[1]]), name='filter-weights1')
                tf.add_to_collection('conv_weights', weights2)
                # Create the convolutional layer; the convolutional layer always us the data as input
                conv_layer2 = tf.nn.conv2d(input=layers[0], filter=weights2, strides=[1, 1, 1, 1], padding=padding_type,
                                          name="conv-layer1")
                layers.append(conv_layer2)
            else:
                for index, number in enumerate(self.filter_number):
                    if self.filter_direction == 'feature sliding':
                        filter_height = self.filter_slide[index]
                        filter_width = self.nt_positions
                    elif self.filter_direction == 'positional sliding':
                        filter_height = self.nt_features
                        filter_width = self.filter_slide[index]
                    weights = tf.Variable(tf.random_normal([filter_height, filter_width, 1, number]),
                                          name='filter-weights')
                    tf.add_to_collection('conv_weights', weights)
                    # Create the convolutional layer; the convolutional layer always us the data as input
                    conv_layer = tf.nn.conv2d(input=layers[0], filter=weights, strides=[1, 1, 1, 1],
                                              padding=padding_type, name="conv-layer{}".format(index))
                    layers.append(conv_layer)

            # Give a defined pooling layer to a specific convolutional layer
            # Axis=1 can be used to pool between feature data
            if self.filter_direction == 'feature sliding':
                axis_number = 1
            # Axis=2 can be used to pool between positional data
            elif self.filter_direction == 'positional sliding':
                axis_number = 2
            # Create a pooling layer if it is defined
            if pooling_type == 'average pooling':
                if self.filter_direction == 'both':
                    print("It is not (yet) possible to pool the convolutional layer if both positional and feature "
                          "sliding is used. Choose in this case for no pooling layer.")
                    exit()
                self.pooling_layer = tf.reduce_mean(input_tensor=layers[-1], axis=axis_number)
                layers.append(self.pooling_layer)
            elif pooling_type == 'max pooling':
                if self.filter_direction == 'both':
                    print("It is not (yet) possible to pool the convolutional layer if both positional and feature "
                          "sliding is used. Choose in this case for no pooling layer.")
                    exit()
                self.pooling_layer = tf.reduce_max(input_tensor=layers[-1], axis=axis_number)
                print(layers[-1])
                layers.append(self.pooling_layer)
            elif pooling_type == 'no pooling':
                pass

            # Check if there are more convolutional layers; these should be combined for the fully connected layer
            layers_to_combine = []
            for layer in layers:
                if 'conv-layer' in layer.name:
                    layers_to_combine.append(layer)

            # TODO: proper combine convolutional layers with a different filter width
            if self.filter_direction == 'both':
                flatten_layers = []
                # Flatten the convolutional layers in a 1D vector
                for layer in layers_to_combine:
                    flatten_layers.append(tf.contrib.layers.flatten(layer))
                # Combine the flatten layer for the fully connected layer
                combined_layer = tf.concat(flatten_layers, axis=-1)
                layers.append(combined_layer)
            else:
                # Flatten the convolutional data in a 1D vector for the fully connected layer
                if len(layers_to_combine) == 1:
                    # Use only the last defined layer for the fully connected layer
                    flatten_conv_layer = tf.contrib.layers.flatten(layers[-1])
                    layers.append(flatten_conv_layer)
                else:
                    # Use all the defined convolutional layers for the fully connected layer
                    combined_layer = tf.concat(layers_to_combine, axis=-1)
                    flatten_conv_layer = tf.contrib.layers.flatten(combined_layer)
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

    def train(self, n_iterations, test_samples, test_labels, training_samples, training_labels, stop_samples,
              stop_labels):
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
        stop_samples: numpy array, contains features scores with shape (number of samples, number of features) of
         the stop validation samples
        stop_labels: numpy array, contains for the stop validation set class label 0 (benign SNPs) and 1 (deleterious
         SNPs) in the shape (number of samples, class dimension)
        """

        # Keep track of the running time for training the neural network
        start_time_network = time.time()

        # Train the neural network with the defined number of iterations on the training data batches
        all_training_loss = []
        all_test_loss = []
        all_stop_loss = []
        mean_training_loss = []
        mean_test_loss = []
        mean_stop_loss = []
        all_training_accuracy = []
        all_test_accuracy = []
        all_training_ROC = []
        all_test_ROC = []
        for iteration in tqdm(range(n_iterations+1)):
            # The dataset create automatic batches, so there is no need to define the samples
            self.session.run(self.optimizer)

            # Check every 50th iteration the loss of the stop set to perform early stopping of the network
            if iteration % 500 == 0:
                training_loss = self.session.run(tf.reduce_mean(self.cost),
                                                 feed_dict={"Input/BatchSamples:0": training_samples,
                                                            "Input/BatchLabels:0": training_labels})
                test_loss = self.session.run(tf.reduce_mean(self.cost),
                                             feed_dict={"Input/BatchSamples:0": test_samples,
                                                        "Input/BatchLabels:0": test_labels})
                stop_loss = self.session.run(tf.reduce_mean(self.cost), feed_dict={"Input/BatchSamples:0": stop_samples,
                                                                                   "Input/BatchLabels:0": stop_labels})
                all_training_loss.append(training_loss)
                all_test_loss.append(test_loss)
                all_stop_loss.append(stop_loss)
                mean_training = np.mean(all_training_loss)
                mean_test = np.mean(all_test_loss)
                mean_stop = np.mean(all_stop_loss)
                mean_training_loss.append(mean_training)
                mean_test_loss.append(mean_test)
                mean_stop_loss.append(mean_stop)
                # Check if the stop loss is increasing or decreasing
                delta_loss = self.best_loss - mean_stop
                print(delta_loss)
                if mean_stop < self.best_loss:
                    self.best_loss = mean_stop
                if delta_loss <= 0.0001:
                    self.best_loss = mean_stop
                    self.stopping_step += 1
                else:
                    self.stopping_step = 0
                # Stop training when the stopping loss is stable five times in a row
                if self.stopping_step == 5:
                    print("Early stopping is triggered at step {} with a loss of {}".format(iteration, mean_stop))
                    break

            # Check for every 500th iteration the loss, and accuracies
            if iteration % 500 == 0:
                training_cost = self.session.run(tf.reduce_mean(self.cost))
                print("STEP {} | Training cost: {:.4f}".format(iteration, training_cost*100))
                test_accuracy = self.evaluate(evaluation_samples=test_samples, evaluation_labels=test_labels)
                print("\t\t   Test accuracy: {:.2f}%".format(test_accuracy[1]))
                training_accuracy = self.evaluate(evaluation_samples=training_samples, evaluation_labels=training_labels)

                all_training_accuracy.append(training_accuracy[1])
                all_test_accuracy.append(test_accuracy[1])
                all_training_ROC.append(training_accuracy[0])
                all_test_ROC.append(test_accuracy[0])

        # Get the total running time of the neural network
        network_run_time = time.time() - start_time_network

        # self.stop_loss = all_stop_loss
        self.stop_loss = mean_stop_loss
        self.all_training_accuracy = all_training_accuracy
        self.all_test_accuracy = all_test_accuracy
        self.all_training_ROC = all_training_ROC
        self.all_test_ROC = all_test_ROC

        # return network_run_time, all_training_loss, all_test_loss
        return network_run_time, mean_training_loss, mean_test_loss

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
                      accuracy_test, accuracy_val, ROC_AUC_train, ROC_AUC_test, ROC_AUC_val, run_time, padding_type,
                      dir, filter_width):
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
        padding_type: string, defines which type of padding is used for the convolutional layer
        dir: string, direction how the filter will slide over the convolutional layer
        filter_width: list:
        """

        # Get the file name where the statistical results will be written to
        # working_dir = os.path.dirname(os.path.abspath(__file__))
        # file_name = working_dir + "/output_ANN/new_norm_ANN.txt"
        file_name = "/mnt/scratch/kersj001/results/ANN_output/statistical_values_ANN.txt"

        # Extend the file with the results in the corresponding data types
        with open(file_name, 'a') as output:
            output.write(
                "\n{:s}\t{:d}\t{:d}\t{:s}\t{:s}\t{:s}\t{:d}\t{:s}\t{:.2f}\t{:f}\t{:d}\t{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t"
                "{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.0f}".format(
                    all_features, n_samples, n_neighbours, layers, padding_type.lower(), dir, filter_width[0],
                    act_func, dropout_rate, learning_rate, iterations, optimum_idx, optimum_loss, start_test_cost,
                    end_test_cost, accuracy_train, accuracy_test, accuracy_val, ROC_AUC_train, ROC_AUC_test,
                    ROC_AUC_val, run_time))
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

        plt.figure()
        plt.plot(training_costs, label="Training loss")
        plt.plot(test_costs, label="Test loss")
        plt.plot(self.stop_loss, label="Stop loss")
        plt.xlabel("Iterations (x500)", size='medium')
        plt.ylabel("Cost function (%)", size='medium')
        plt.suptitle("Mean loss while training the neural network", size='medium', ha='center')
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
        # plt.ylim([0.30, 0.50])
        plt.legend(loc='right', bbox_to_anchor=(1.39, 0.5))
        plt.subplots_adjust(right=0.75)
        saving("/mnt/scratch/kersj001/results/ANN_output/loss_plots/{}_error_{}".format(n_neighbours, data_size))

    def accuracy_graph(self, learning_rate, training_accuracy, test_accuracy, val_accuracy, layers, data_size,
                       n_neighbours, dropout_layer, dropout_rate):
        """ Return plot of the decreasing error during the training of the neural network

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

        plt.figure()
        plt.plot(self.all_training_accuracy, label="Training accuracy")
        plt.plot(self.all_test_accuracy, label="Test accuracy")
        plt.plot(self.all_training_ROC, label="Training ROC AUC")
        plt.plot(self.all_test_ROC, label="Test ROC AUC", color='tab:gray')
        plt.xlabel("Iterations (x500)", size='medium')
        plt.ylabel("Accuracy (%)", size='medium')
        plt.suptitle("Accuracy while training the neural network", size='medium', ha='center')
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
        # plt.ylim([0.35, 0.50])
        plt.legend(loc='right', bbox_to_anchor=(1.48, 0.55))
        plt.subplots_adjust(right=0.75)
        saving("/mnt/scratch/kersj001/results/ANN_output/accuracy_plots/{}_accuracy_{}".format(n_neighbours, data_size))

    def plot_conv_weights(self, weight, training, test, val, roc_training, roc_test, roc_val, plot_orientation,
                          subplot_number=""):
        """ Create .png file the desired directory containing absolute weight values of each filter

        weight: numpy array of shape (filter length, filter width, channel, number of filters)
        training: float, defines the training accuracy after the neural network training
        test: float, defines the test accuracy after the neural network training
        val: float, defines the validation accuracy after the neural network training
        roc_training: float, defines the training ROC AUC score after the neural network training
        roc_test: float, defines the test ROC AUC score after the neural network training
        roc_val: float, defines the validation ROC AUC score after the neural network training
        plot_orientation: string, defines is the plot will be landscape or portrait
        subplot_number: integer, defines if the subplot number if more convolutional layers are used, default=""
        """

        # Convert the weight values to absolute values
        weight = np.absolute(weight)

        # Get the minimum and maximum weight for defining the axis
        min_weight = np.min(weight)
        max_weight = np.max(weight)

        # Get the number of convolutional filters
        n_filters = weight.shape[3]

        # Create a figure with for each filter a subplot
        if plot_orientation == 'vertical':
            total_fig_width = int(((n_filters/2) * self.filter_slide[0]) + 1)  # +1 for the colorbar
            fig, axes = plt.subplots(nrows=1, ncols=n_filters, figsize=(total_fig_width, 3))
            color_bar_orientation = 'vertical'
        elif plot_orientation == 'horizontal':
            total_fig_height = int((n_filters/2) * self.filter_slide[0])
            if total_fig_height < 1:
                total_fig_height = 1
            fig, axes = plt.subplots(nrows=n_filters, ncols=1, figsize=(3, total_fig_height))
            color_bar_orientation = 'horizontal'
        # Iterate over the filters
        if n_filters > 1:
            for n_filter, ax in enumerate(axes.flat):
                # Get a single filter
                plot = weight[:, :, 0, n_filter]
                # Put the filter plot on the grid
                im = ax.imshow(X=plot, vmin=min_weight, vmax=max_weight, interpolation='nearest', cmap='Reds')
                # Do not set the axis labels for each subplot, except for the y-labels of the first plot
                ax.set_xticks([])
                ax.set_yticks([])
            # Add the colorbar
            plt.colorbar(im, ax=axes.ravel().tolist(), pad=0.05, orientation=color_bar_orientation)
        else:
            # Get a single filter
            plot = weight[:, :, 0, 0]
            # Put the filter plot on the grid
            im = axes.imshow(X=plot, vmin=min_weight, vmax=max_weight, interpolation='nearest', cmap='Reds')
            # Do not set the axis labels for each subplot, except for the y-labels of the first plot
            axes.set_xticks([])
            axes.set_yticks([])
            # Add the colorbar
            plt.colorbar(im, ax=axes, pad=0.05, orientation=color_bar_orientation)
        # Add the end accuracy and ROC AUC scores to the plot
        fig_size_cor = n_filters / 2
        if fig_size_cor < 1:
            fig_size_cor = 1
        if plot_orientation == 'vertical':
            plt.figtext(1.30, 0.75, "Training accuracy\n{0:.2f}%".format(training), size='small')
            plt.figtext(1.30, 0.62, "Test accuracy\n{0:.2f}%".format(test), size='small')
            plt.figtext(1.30, 0.49, "Validation accuracy\n{0:.2f}%".format(val), size='small')
            plt.figtext(1.30, 0.36, "Training ROC AUC\n{0:.2f}%".format(roc_training), size='small')
            plt.figtext(1.30, 0.23, "Test ROC AUC\n{0:.2f}%".format(roc_test), size='small')
            plt.figtext(1.30, 0.10, "Validation ROC AUC\n{0:.2f}%".format(roc_val), size='small')
        elif plot_orientation == 'horizontal':
            plt.figtext(1.00, 1.00/fig_size_cor, "Training accuracy: {0:.2f}%".format(training), size='small')
            plt.figtext(1.00, 0.83/fig_size_cor, "Test accuracy: {0:.2f}%".format(test), size='small')
            plt.figtext(1.00, 0.60/fig_size_cor, "Validation accuracy: {0:.2f}%".format(val), size='small')
            plt.figtext(1.00, 0.40/fig_size_cor, "Training ROC AUC: {0:.2f}%".format(roc_training), size='small')
            plt.figtext(1.00, 0.20/fig_size_cor, "Test ROC AUC: {0:.2f}%".format(roc_test), size='small')
            plt.figtext(1.00, 0.01/fig_size_cor, "Validation ROC AUC: {0:.2f}%".format(roc_val), size='small')

        # Save the figure
        # Create the name for saving which contains the number of used filters of each layer
        name_filter_number = list(map(lambda number: str(number), self.filter_number))
        name_filter_number = '-'.join(name_filter_number)
        # Get the directory for saving the file
        saving("/mnt/scratch/kersj001/results/ANN_output/weight_plots/{}_ConvWeights_{}-filters_{}".format(n_neighbours,
               name_filter_number, data_size), extension='{}'.format(subplot_number))


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


def define_parameters(all_features, act_func, dropout, fc_layer_units, training_samples, batch_perc, pooling,
                      filter_units, width_filter, sliding_dir, padding):
    """ Convert some parameters in the right format

    act_funct: string, defines which activation function will  be used for training the neural network
    dropout: string, contains boolean values that defines if a fully connected layer is followed by a dropout layer
    fc_layers_units: string, contains integer values that defines the number of nodes for the fully connected layers
    training_samples: numpy array, contains features scores with shape (samples, number of features * number of
     neighbouring positions) of the training samples
    batch_perc: flaot, defines the percentage of the training samples that will be used as batch size
    pooling: string, refers to the type of pooling layer
    filter_units: string, contains integer values that defines the number of filter for the convolutional layers
    width_filter: string, contains integer values that defines the filter width for the convolutional layers
    sliding_dir: string: defines in what direction the filter will slide over the convolutional layer
    padding: string, defines which padding type for the convolutional layer is used
    """

    # Get the variables in the right format
    all_features = all_features.lower()
    if all_features == 'y' or all_features == 'yes':
        all_features = "all conservation-based scores"
    else:
        all_features = "only PhastCon primate scores"

    # Get a specific name for the filter sliding direction
    sliding_dir = sliding_dir.lower()
    if sliding_dir == 'h' or sliding_dir == 'hor' or sliding_dir == 'horizontal' \
            or sliding_dir == 'f' or sliding_dir == 'feature':
        sliding_dir = 'feature sliding'
    elif sliding_dir == 'p' or sliding_dir == 'pos' or sliding_dir == 'v' or sliding_dir == 'ver':
        sliding_dir = 'positional sliding'
    else:
        sliding_dir = 'both'

    padding = padding.lower()
    if padding == 's' or padding == 'same':
        padding = 'SAME'
    else:
        padding = 'VALID'

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

    # Check if the number of hidden layer nodes is equal to the defined drop out layers
    if len(int_layer_units) != len(dropout_booleans):
        print("The number of drop out layers is not equal to the number of hidden layers")
        exit()

    # Get the number of used layers in correct format
    filters = filter_units.split(',')
    filter_units = list(map(lambda filter: int(filter), filters))

    # Get the filter width of the convolutional layers
    widths = width_filter.split(',')
    width_filter = list(map(lambda width: int(width), widths))

    # Check if the number of defined convolutional layers is equal to number of the defined filter sizes
    if len(filter_units) != len(width_filter):
        print("The number of defined filter sizes is not equal to the number of defined convolutional layers")
        exit()

    # Get the convolutional layer names of the neural network architecture
    layers = []
    for index, unit in enumerate(filter_units):
        layers.append('conv (14x{}, {})'.format(width_filter[index], unit))

    # Get first the pooling layer type
    pooling = pooling.lower()
    if pooling == 'average' or pooling == 'avg' or pooling == 'a' or pooling == 'mean':
        pooling_name = 'average pooling'
        layers.append('avg')
    elif pooling == 'max' or pooling == 'm':
        pooling_name = 'max pooling'
        layers.append('max')
    else:
        pooling_name = 'no pooling'

    # Append the layer names with the fully connected and dropout layers
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

    return all_features, act_func, act_title, batch_size, layers, dropout_booleans, int_layer_units, pooling_name, \
           filter_units, width_filter, sliding_dir, padding


def get_arguments():
    """ Return the arguments given on the command line
    """

    # # Specify the variable if you do not run from the command line
    # # data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/2_200000.h5"
    # # val_data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/2_26172.h5"
    # data_directory = "/mnt/nexenta/kersj001/data/normalized/200_thousand/2_200000.h5"
    # val_data_directory = "/mnt/nexenta/kersj001/data/normalized/200_thousand/2_26172.h5"
    # all = 'yes'
    # snp_neighbour = 'neighbour'
    # # snp_neighbour = 'snp'
    # learning_rate = 0.01
    # dropout_rate = 0.5
    # activation_function = 't'
    # batch_percentage = 0.01
    # iterations = 500000
    # fc_nodes = '15'
    # dropout_layer = 'f'
    # pooling_layer = 'n'
    # n_filter = '3'
    # filter_size = '5'
    # sliding = 'p'
    # padding_use = 'v'

    # Specify the options for running from the command line
    parser = OptionParser()
    # Specify the data directory for the benign and deleterious SNPs
    parser.add_option("-d", "--data", dest="data", help="Path to the output of the normalized feature scores of \
        deleterious SNPs and its neighbouring features", default="")
    # Specify the data directory for the validation samples
    parser.add_option("-v", "--valdata", dest="validation_data", help="Path to the normalized validation samples",
        default="")
    # Specify if only the PhastCon primate scores are considered or all the conservation-based features
    parser.add_option("-f", "--all", dest="all_features", help="String (yes or no) that indicates if all conservation-\
        based features are considered", default="yes")
    # Specify if the neural network will included neighbouring positions
    parser.add_option("-s", "--snp", dest="snp_neighbour", help="String that indicates if the surrounding neighbours \
        will be included ('n') or excluded ('s')", default="n")
    # Specify the learning rate
    parser.add_option("-l", "--learning", dest="learning_rate", help="Float that defines the learning rate of for \
        training the neural network", default=0.01)
    # Specify the dropout rate
    parser.add_option("-r", "--droprate", dest="dropout_rate", help="Float that defines the dropout rate for the \
        dropout layers", default=0.5)
    # Specify the activation function
    parser.add_option("-a", "--act", dest="activation", help="String that refers to the activation function that will \
        be used for training the neural network", default="relu")
    # Specifiy the number of iterations
    parser.add_option("-i", "--iter", dest="iterations", help="Integer that defines the number of iterations for \
        training the neural network", default=1000)
    # Specify the percentage of training data that will be use as batch size
    parser.add_option("-b", "--batch", dest="batch_perc", help="Float that defines the percentage of training samples \
        that will be used as batch size", default=0.01)
    # Specify the nodes for the fully connected layers
    parser.add_option("-n", "--nodes", dest="fc_nodes", help="List that contains integer values that defines the \
        number of nodes for the fully connected layer", default="4")
    # Specify if dropout layers occurs after a fully connected layer
    parser.add_option("-o", "--dropout", dest="dropout_layers", help="List that contains boolean values that defines \
        if a fully connected layer is followed by a dropout layer", default="False")

    parser.add_option("-p", "--pool", dest="pooling_layer", help="String that refers if and what type of pooling \
        layer is used", default="none")
    parser.add_option("-u", "--filter", dest="n_filter", help="List that contains integer values that defines the \
        number of filter for every convolutional layer", default="1")
    parser.add_option("-w", "--width", dest="filter_size", help="List that contains integer values that defines the \
        filter width for every convolutional layer", default="1")
    parser.add_option("-x", "--slice", dest="sliding", help="String that defines in what direction the filter will \
        slide over the convolutional layer", default="1")
    parser.add_option("-y", "--padding", dest="padding", help="String that defines in which type of padding will be \
        for the convolutional layers", default="valid")

    # Get the command line options for reading the data for both the benign and deleterious SNPs
    (options, args) = parser.parse_args()
    data_directory = options.data
    val_data_directory = options.validation_data
    all = options.all_features
    snp_neighbour = options.snp_neighbour
    learning_rate = float(options.learning_rate)
    dropout_rate = float(options.dropout_rate)
    activation_function = options.activation
    batch_percentage = float(options.batch_perc)
    iterations = int(options.iterations)
    fc_nodes = options.fc_nodes
    dropout_layer = options.dropout_layers
    pooling_layer = options.pooling_layer
    n_filter = options.n_filter
    filter_size = options.filter_size
    sliding = options.sliding
    padding_use = options.padding

    return data_directory, val_data_directory, all, snp_neighbour, learning_rate, dropout_rate, activation_function, \
           batch_percentage, iterations, fc_nodes, dropout_layer, pooling_layer, n_filter, filter_size, sliding, \
           padding_use


if __name__ == "__main__":
    # Keep track of the running time
    start_time_script = time.time()

    # Get the given arguments including the hyperparameters for the neural network
    data_directory, val_data_directory, all_features, snp_neighbour, learning_rate, dropout_rate, \
        activation_function_name, batch_percentage, iterations, fc_nodes, dropout_layers, pooling_layer_type, \
        n_filter, filter_size, sliding, used_padding = get_arguments()

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

    # Define a training (60%), test (20%), and stop (20%) set
    training_data, test_data, training_labels, test_labels = train_test_split(samples, labels, test_size=0.4)
    test_data, stop_data, test_labels, stop_labels = train_test_split(test_data, test_labels, test_size=0.5)

    # Get the parameters for the neural network in correct format
    feature_title, act_func, act_title, batch_size, layer_names, dropout_booleans, fc_layer_units, pooling_layer_name, \
        n_filters, filter_size, sliding_dir, padding = define_parameters(all_features=all_features,
        act_func=activation_function_name, dropout=dropout_layers, fc_layer_units=fc_nodes,
        training_samples=training_data, batch_perc=batch_percentage, pooling=pooling_layer_type, filter_units=n_filter,
        width_filter=filter_size, sliding_dir=sliding, padding=used_padding)

    # Create the neural network
    nn = NeuralNetwork(all_samples=training_data, all_labels=training_labels, fc_layers=fc_layer_units,
                       do_layers=dropout_booleans, dropout_rate=dropout_rate, act_func=act_func,
                       learning_rate=learning_rate, batch_size=batch_size, feature_number=n_features,
                       position_number=n_positions, pooling_type=pooling_layer_name, filter_number=n_filters,
                       filter_slide=filter_size, filter_direction=sliding_dir, padding_type=padding)

    # Train the neural network
    run_time, training_loss, test_loss = nn.train(n_iterations=iterations, test_samples=test_data,
                                                  test_labels=test_labels, training_samples=training_data,
                                                  training_labels=training_labels, stop_samples=stop_data,
                                                  stop_labels=stop_labels)

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
        min_test_cost_idx = test_loss.index(min_test_cost) * 500
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
                     run_time=run_time, padding_type=padding, dir=sliding_dir, filter_width=filter_size)

    # Create a plot of the loss function
    nn.loss_graph(training_costs=training_loss, test_costs=test_loss, learning_rate=learning_rate, training_accuracy=
                  training_accuracy, test_accuracy=test_accuracy, val_accuracy=val_accuracy, layers=layer_names,
                  data_size=data_size, n_neighbours=n_neighbours, dropout_layer=dropout_booleans, dropout_rate=
                  dropout_rate)

    # Create a plot of the loss function
    nn.accuracy_graph(learning_rate=learning_rate, training_accuracy=training_accuracy, test_accuracy=test_accuracy,
                      val_accuracy=val_accuracy, layers=layer_names, data_size=data_size, n_neighbours=n_neighbours,
                      dropout_layer=dropout_booleans, dropout_rate=dropout_rate)

    # Get the filter weights for every convolutional layer
    end_weights = nn.session.run([tf.get_collection('conv_weights')])
    # The weight are located at position 0 of the array
    for index, weight in enumerate(end_weights[0]):
        # Define the plot orientation
        if sliding_dir == 'feature sliding':
            plot_format = 'horizontal'
        elif sliding_dir == 'positional sliding':
            plot_format = 'vertical'
        else:
            # The first convolutional layer slices positional if both directions are define,
            if index == 0:
                plot_format = 'vertical'
            else:
                plot_format = 'horizontal'
        if len(end_weights[0]) > 1:
            nn.plot_conv_weights(weight, training_accuracy, test_accuracy, val_accuracy, training_roc, test_roc,
                                 val_roc, plot_format, subplot_number='-'+str(index+1))
        else:
            nn.plot_conv_weights(weight, training_accuracy, test_accuracy, val_accuracy, training_roc, test_roc,
                                 val_roc, plot_format)

    # Get the complete running time of the script
    print("----- {} seconds -----".format(round(time.time() - start_time_script), 2))
