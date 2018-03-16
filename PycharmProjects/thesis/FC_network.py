#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date of creation: 16 January 2018
Date of last edit: 15 March 2018
Script for performing neural network on the SNP data set

Output are plots to show the loss and accuracy and .h5 file with the weights for each feature
"""


import time
import h5py
import os
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from optparse import OptionParser


def shuffled_copies(samples, labels):
    """ Return a shuffled array with samples and labels

    samples: numpy array, contains features
    labels: one hot encoded array, contains data labels
    """

    # Check if the samples and labels are from the same format
    assert len(samples) == len(labels)
    permu = np.random.permutation(len(samples))

    # Get the correct indexes
    samples = samples[permu]
    labels = labels[permu]

    return samples, labels


def neural_network(vec, labels, val_vec, val_labels, data_size, n_neighbours, learning_rate):
    """ Return predicted labels and actual labels with error rate and accuracy after training the fully connected
    neural network

    vec: numpy array, contains features scores with shape (samples, number of features * number of neighbouring
     positions)
    labels: one hot encoded numpy array, contains data labels
    val_vec: numpy array, contains features scores of the validation samples with shape (samples, number of features *
     number of neighbouring positions)
    val_labels: one hot encoded numpy array, contains validation data labels
    data_size: integer, defines the number of samples in the data
    n_neighbours: integer, indicates how many neighbours are included
    learning_rate: float, defines the learning rate for training the network
    """

    # HYPER PARAMETERS
    act_func = tf.nn.relu
    pred_act_func = tf.nn.softmax
    dropout = False
    layer_1_units = 5
    layer_2_units = 7
    layer_3_units = 5
    layer_units = 5 #(5, 7, 5)
    number_hidden_layers = 1
    # number_hidden_layers = len(layer_units)
    iterations = 50
    graph_offset = -2
    dropout_rate = 0.5

    # DATA HANDLING
    # Define training and test set
    test_size = 0.1  # training is set on 80%
    training_vec, test_vec, training_labels, test_labels = train_test_split(vec, labels, test_size=test_size)
    batch_percentage = 0.1 # There is chosen to use a batch size of 10%
    batch_size = int(training_vec.shape[0] * batch_percentage)
    feature_number = training_vec.shape[1]
    class_number = len(np.unique(labels))

    # GRAPH DEFINITION
    # Use scope to get a nicely lay-out in TensorBoard
    with tf.variable_scope("input"):
        # Define a placeholder for both the data and labels
        data_input = tf.placeholder(tf.float32, shape=[None, feature_number], name="data_input")
        label_input = tf.placeholder(tf.float32, shape=[None, class_number], name="label_input")

    # Fully-connected layers
    with tf.variable_scope("layers"):
        fully_connected_1 = tf.layers.dense(inputs=data_input, units=layer_1_units, activation=act_func, use_bias=True,
                                            name="fc-layer1")
        # drop_out_1 = tf.layers.dropout(inputs=fully_connected_1, rate=dropout_rate, name="drop-out1")
        # fully_connected_2 = tf.layers.dense(inputs=fully_connected_1, units=layer_2_units, activation=act_func,
        #                                     use_bias=True, name="fc-layer2")
        # fully_connected_2 = tf.layers.dense(inputs=drop_out_1, units=layer_2_units, activation=act_func,
        #                                     use_bias=True, name="fc-layer2")
        # drop_out_2 = tf.layers.dropout(inputs=fully_connected_2, rate=dropout_rate, name="drop-out2")
        # fully_connected_3 = tf.layers.dense(inputs=fully_connected_2, units=layer_3_units, activation=act_func,
        #                                     use_bias=True, name="fc-layer3")
        # fully_connected_3 = tf.layers.dense(inputs=drop_out_2, units=layer_3_units, activation=act_func,
        #                                     use_bias=True, name="fc-layer3")
        # drop_out_3 = tf.layers.dropout(inputs=fully_connected_3, rate=dropout_rate, name="drop-out3")

    with tf.variable_scope('output'):
        pred = tf.layers.dense(inputs=fully_connected_1, units=2, activation=pred_act_func, use_bias=True,
                               name="prediction")
        # pred = tf.layers.dense(inputs=drop_out_1, units=2, activation=pred_act_func, use_bias=True,
        #                        name="prediction")

    with tf.variable_scope("prediction"):
        # Normalize the cost for the batch size (denominator) with euclidean distance
        # cost = tf.nn.l2_loss(pred - label_input)
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label_input)
        cost = tf.reduce_mean(cost) * 100

    with tf.variable_scope("optimizer"):
        # minimization_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        minimization_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./graphs/tensorboard", graph=sess.graph)

    # CHECKS
    # Set the range for looping to train the neural network
    all_costs = []
    all_test_costs = []
    all_accuracy = [0]
    all_test_accuracy = [0]
    for i in range(iterations+1):
        # To prevent slicing will be out of range
        offset = int(i * batch_size % len(training_vec))
        # Epoch wise training data shuffling
        if offset + batch_size >= len(training_vec):
            training_vec, training_labels = shuffled_copies(training_vec, training_labels)
            test_vec, test_labels = shuffled_copies(test_vec, test_labels)

        # Get the accuracy for the first 20 iteration, because this is a steep part in the graph
        # if i in range(1, graph_offset+1):
        #     accuracy_sess = get_accuracy(data_input, label_input, pred, sess, training_labels, training_vec)
        # all_accuracy.append((accuracy_sess * 100))

        # Thereafter get the accuracy for every 5th iteration
        if i in range(graph_offset+2, iterations+1) and i % 5 == 0:
            accuracy_sess = get_accuracy(data_input, label_input, pred, sess, training_labels, training_vec)
            accuracy_sess_test = get_accuracy(data_input, label_input, pred, sess, test_labels, test_vec)
        # all_accuracy.append((accuracy_sess * 100))

        # Check every 50th loop how well the prediction is
        if i % 50 == 0:
            # pred_out, label_input_out = sess.run([pred, label_input],
            #                                      feed_dict={data_input: training_vec[offset:offset + batch_size],
            #                                                 label_input: training_labels[offset:offset + batch_size]})
            # # Compare the true labels with the predicted labels
            # correct_prediction = tf.equal(tf.argmax(pred_out, axis=1), tf.argmax(label_input_out, axis=1))
            #
            # # Check the accuracy by counting the miss-classifications
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # accuracy_sess = sess.run(accuracy, feed_dict={label_input: test_labels})
            accuracy_sess = get_accuracy(data_input, label_input, pred, sess, training_labels, training_vec)
            print("accuracy   :  {0:.2f}".format(accuracy_sess), '%\n')
            # all_accuracy.append(accuracy_sess)

        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # accuracy_sess = sess.run(accuracy, feed_dict={label_input: test_labels})

        _, current_cost = sess.run([minimization_step, cost],
                                   # feed_dict={data_input: training_vec[offset:offset + batch_size],
                                   #            label_input: training_labels[offset:offset + batch_size]})
                                   feed_dict = {data_input: training_vec,
                                                label_input: training_labels})
        test_cost = sess.run([cost], feed_dict={data_input: test_vec,
                                                label_input: test_labels})

        # Check every 10th loop what the cost is
        if i % 10 == 0:
            print("STEP: {} | Average cost: {}".format(i, np.mean(current_cost)))

        all_costs.append(current_cost)
        all_test_costs.append(test_cost)
        all_accuracy.append((accuracy_sess))
        all_test_accuracy.append((accuracy_sess_test))

    test_cost = sess.run(cost, feed_dict={data_input: test_vec,
                                          label_input: test_labels})
    val_cost = sess.run(cost, feed_dict={data_input: val_vec,
                                         label_input: val_labels})
    pred_val, labels_val = sess.run([pred, label_input],
                                   feed_dict={data_input: val_vec,
                                              label_input: val_labels})
    # ROC_val = tf.metrics.auc(labels=labels_val, predictions=pred_val)

    end_test_accuracy = get_accuracy(data_input, label_input, pred, sess, test_labels, test_vec)
    end_val_accuracy = get_accuracy(data_input, label_input, pred, sess, val_labels, val_vec)

    print("\nend accuracy of the predictions: {0:.2f}%".format(accuracy_sess))
    print("test accuracy: {0:.2f}%".format(end_test_accuracy))
    print("validation accuracy: {0:.2f}%".format(end_val_accuracy))
    print("\ntest_cost: {} ".format(test_cost))
    print("validation_cost: {} ".format(val_cost))
    # print("ROC-AUC validation: {}".format(ROC_val))

    min_test_cost = min(all_test_costs)
    mim_test_cost_idx = all_test_costs.index(min_test_cost)
    print("\noptimum number of iterations: ", mim_test_cost_idx)

    # Get the weights after training
    influence_weights(sess, n_neighbours, data_size)

    # Create a plot of the cost function
    error_plot(all_costs, all_test_costs, number_hidden_layers, learning_rate, accuracy_sess, end_test_accuracy,
               end_val_accuracy, layer_units, data_size, n_neighbours, min_test_cost[0], mim_test_cost_idx,
               y_axis="Cost function", dropout=dropout)

    accuracy_plot(all_accuracy, all_test_accuracy, n_neighbours, data_size, number_hidden_layers, layer_units,
                  learning_rate, dropout=dropout)
    # Close the TensorFlow session
    writer.close()
    sess.close()

    return all_costs


def get_accuracy(data_input, label_input, pred, sess, labels, samples):
    """ Get the accuracy while training the neural network

    data_input: TensorFlow placeholder, contains a defined subpart of the data samples
    label_input: TensorFlow placeholder, contains a defined subpart of the data labels
    pred: tensor, contains predicted class labels
    sess: TensorFlow object, encapsulates the environment in which operation objects are executed and Tensor objects
     are evaluated
    labels: one hot encoded numpy array, contains data labels
    samples: numpy array, contains features scores with shape (samples, number of features * number of neighbouring
     positions)
    """

    # Get the predicted labels and the real labels
    pred_out, label_input_out = sess.run([pred, label_input],
                                         # feed_dict={data_input: training_vec[offset:offset + batch_size],
                                         #            label_input: training_labels[offset:offset + batch_size]})
                                         feed_dict={data_input: samples,
                                                    label_input: labels})

    # Compare the true labels with the predicted labels
    correct_prediction = tf.equal(tf.argmax(pred_out, axis=1), tf.argmax(label_input_out, axis=1))
    # Check the accuracy by counting the miss-classifications
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_sess = sess.run(accuracy, feed_dict={label_input: labels})

    return accuracy_sess*100


def influence_weights(sess, n_neighbours, data_size):
    """ Returns the weights for each feature after training the neural network in a numpy array

    sess: TensorFlow object, encapsulates the environment in which operation objects are executed and Tensor objects
     are evaluated
    n_neighbours: integer, indicates how many neighbours are included
    data_size: integer, defines the number of samples in the data
    """

    # To see the names of the different layers of the neural network
    tf.global_variables()

    # Get the weights for a specified layer
    layer_number = 1
    weights = tf.get_default_graph().get_tensor_by_name('layers/fc-layer{:d}/kernel:0'.format(layer_number))
    weights = sess.run(tf.nn.top_k(weights))
    # nn.top_k has as output [[weights][indexes to next node]], so the weights are located at position 0
    weights_numpy = np.array(weights[0])

    # Get the working directory of this script
    working_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the file name for saving the numpy array
    file_name = working_dir + "/output_ANN/HDF5_files/normalized_ANN/{}_norm_ANNweights_{}".format(
        n_neighbours, data_size)
    # Get an unique file name
    file_name = saving(file_name, save=False, file_type='h5')
    # Save the numpy array as HDF5 file
    h5f = h5py.File(file_name, 'w')
    h5f.create_dataset('dataset_{}'.format(data_size), data=weights_numpy)

    return weights_numpy


def error_plot(training_costs, test_costs, n_hidden_layers, learning_rate, accuracy, test_accuracy, val_accuracy,
               layer_units, data_size, n_neighbours, optimum, optimum_idx, y_axis, dropout=False):
    """ Return plot of the decreasing error during the training of the neural network

    training_costs: list, contains training error costs
    test_costs: list, contains test error costs
    n_hidden_layers: integer, number of hidden layers
    learning_rate: float, strictness of learning while training the neural network
    accuracy: float, defines the end training accuracy after training the neural network
    test_accuracy: float, defines the end test accuracy after training the neural network
    val_accuracy: float, defines the end validation accuracy after training the neural network
    layer_units: tuple, contains integers which indicated how many nodes are used for each hidden layer
    data_size: integer, defines the number of samples in the data
    n_neighbours: integer, indicates how many neighbours are included
    optimum: float, gives the minimum loss
    optimum_idx: integer, gives the index value of the minimum loss
    y_axis: string, defines the name of the y-axis of the plot
    drop_out: boolean, indicates if there is a dropout layer included, default=False
    """

    plt.plot(training_costs, label="Training loss")
    plt.plot(test_costs, label="Test loss")
    plt.xlabel("Iterations")
    plt.ylabel(y_axis)
    if dropout == True:
        plt.title("Cost function while training the neural network \n{} hidden layer(s) - {} and a drop out layer,"
                  " learning rate: {}".format(n_hidden_layers, layer_units, learning_rate))
    else:
        plt.title("Cost function while training the neural network \n{} hidden layer(s) - {}, learning rate: {}".format(
            n_hidden_layers, layer_units, learning_rate))
    # plt.annotate('({:d},{:.2f})'.format(optimum_idx, optimum), xy=(optimum_idx, optimum))
    plt.figtext(0.77, 0.35, "Training accuracy\n{0:.2f}%".format(accuracy))
    plt.figtext(0.77, 0.25, "Test accuracy\n{0:.2f}%".format(test_accuracy))
    plt.figtext(0.77, 0.15, "Validation accuracy\n{0:.2f}%".format(val_accuracy))
    if n_neighbours == 0:
        plt.figtext(0.77, 0.80, "Neighbours\nexcluded")
    else:
        plt.figtext(0.77, 0.80, "Neighbours\nincluded")
    plt.legend(loc='right', bbox_to_anchor=(1.39, 0.5))
    plt.subplots_adjust(right=0.75)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    saving(working_dir + "/output_ANN/error_plots/{}_error_{}".format(n_neighbours, data_size))


def accuracy_plot(training, test, n_neighbours, data_size, n_hidden_layers, layer_units, learning_rate, dropout=False):
    """ Return plot of the accuracy during the training of the neural network

    training: list, contains training accuracy while training the neural network
    test: list, contains test accuracy while training the neural network
    n_neighbours: integer, indicates how many neighbours are included
    data_size: integer, defines the number of samples in the data
    n_hidden_layers: integer, number of hidden layers
    layer_units: tuple, contains integers which indicated how many nodes are used for each hidden layer
    learning_rate: float, strictness of learning while training the neural network
    drop_out: boolean, indicates if there is a dropout layer included, default=False
    """

    plt.figure()
    plt.plot(training, label="Training")
    plt.plot(test, label="Test")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy function (%)")
    if dropout == True:
        plt.title("Accuracy function while training the neural network \n{} hidden layer(s) - {} and a drop out "
                  "layer, learning rate: {}".format(n_hidden_layers, layer_units, learning_rate))
    else:
        plt.title("Accuracy function while training the neural network \n{} hidden layer(s) - {}, learning rate: {}"
                  "".format(n_hidden_layers, layer_units, learning_rate))
    if n_neighbours == 0:
        plt.figtext(0.83, 0.80, "Neighbours\nexcluded")
    else:
        plt.figtext(0.83, 0.80, "Neighbours\nincluded")
    plt.legend(loc='right', bbox_to_anchor=(1.3, 0.5))
    plt.subplots_adjust(right=0.8)

    working_dir = os.path.dirname(os.path.abspath(__file__))
    saving(working_dir + "/output_ANN/accuracy_plots/{}_accuracy_{}".format(n_neighbours, data_size))


def saving(file_path, save=True, file_type='png'):
    """ Saves the plot in the correct directory with an unique name

    file_path: string, path direction to where the plot will be saved
    """

    index_saving = 1
    while os.path.exists(file_path + "_{}.{}".format(index_saving, file_type)):
        index_saving += 1

    if save == True:
        plt.savefig(file_path + "_{}.{}".format(index_saving, file_type))
    else:
        return file_path + "_{}.{}".format(index_saving, file_type)


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

    # Get the features of the SNP of interest and neighbouring positions
    else:
        # The data should fit in a 2D array for performing neural network. The number of samples should be stay, and
        # the number of features will be the number of features times the number of nucleotides
        samples = data.reshape([data.shape[0], -1])

    return samples


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

    # Convert the data into one hot encoded data
    labels = initialization_based(labels)

    return labels


def initialization_based(input_array):
    """ Return the data as one hot encoded data

    input_array: numpy array, labels of the input data
    """

    # Search for the unique labels in the array
    oh_array = np.unique(input_array, return_inverse=True)[1]

    # Define the shape of the one hot encoded array
    out = np.zeros((oh_array.shape[0], oh_array.max() + 1), dtype=int)

    # Set the predicted class on 1, and all the other classes stays at 0
    out[np.arange(out.shape[0]), oh_array] = 1

    return out


def get_arguments():
    """ Return the arguments given on the command line
    """

    # If you do not run from the command line

    data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/5_200000.h5"
    val_data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/5_26172.h5"
    learning_rate = 0.001
    snp_neighbour = 'neighbour'

    # # Read the data
    # # Specify the options for running from the command line
    # parser = OptionParser()
    # # Specify the data directory for the benign and deleterious SNPs
    # parser.add_option("-d", "--data", dest="data", help="Path to the output of the normalized feature scores of \
    #     deleterious SNPs and its neighbouring features", default="")
    # # Specify the data directory for the validation samples
    # parser.add_option("-v", "--valdata", dest="validation_data", help="Path to the normalized validation samples",
    #     default="")
    # # Specify the learning rate
    # parser.add_option("-l", "--learning", dest="learning_rate", help="Float that defines the learning rate of for \
    #         training the neural network", default=0.01)
    # parser.add_option("-s", "--snp", dest="snp_neighbour", help="String that indicates if the surrounding neighbours \
    #             will be included ('n') or excluded ('s')", default="n")
    #
    # # Get the command line options for reading the data for both the benign and deleterious SNPs
    # (options, args) = parser.parse_args()
    # data_directory = options.data
    # val_data_directory = options.validation_data
    # learning_rate = float(options.learning_rate)
    # snp_neighbour = options.snp_neighbour
    #
    return data_directory, val_data_directory, learning_rate, snp_neighbour


if __name__ == "__main__":
    # Keep track of the running time
    start_time = time.time()

    # Get the given arguments
    data_directory, val_data_directory, learning_rate, snp_neighbour = get_arguments()

    # Read the HDF5 file back to a numpy array
    data, data_size, n_neighbours = data_reading(data_directory)
    val_data, _, _ = data_reading(val_data_directory)

    # Get the number of considered neighbouring positions
    snp_neighbour = snp_neighbour.lower()
    if snp_neighbour == "s" or snp_neighbour == "snp":
        n_neighbours = 0
    else:
        n_neighbours = n_neighbours

    # Parse the data into samples which either consider neighbouring positions or not
    samples = data_parser(data, n_neighbours)
    samples_val = data_parser(val_data, n_neighbours)

    # Get the data labels
    labels = data_labels(data)
    labels_val = data_labels(val_data)

    # Run the neural network
    neural_network(samples, labels, samples_val, labels_val, data_size, n_neighbours, learning_rate)

    print("\n----- running time: {} seconds -----".format(round(time.time() - start_time), 2))
