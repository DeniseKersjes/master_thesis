#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date of creation: 16 January 2018
Date of last edit: 23 March 2018
Script for performing neural network on the SNP data set

Output are plots to show the loss and accuracy and .h5 file with the weights for each feature
"""


import time
import h5py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from optparse import OptionParser


def neural_network(vec, labels, val_vec, val_labels, data_size, n_neighbours, learning_rate, all_features,
                   iterations, act_func, fc_layer_units, dropout, dropout_rate):
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
    all_features: string, indicates if all conservation-based features are considered or only the PhastCon primate
     scores
    iterations: integer, defines the number of iteration for training the neural network
    act_funct: string, defines which activation function will  be used for training the neural network
    fc_layers_units: list, contains integer values that defines the number of nodes for the fully connected layers
    dropout: list, contains boolean values that defines if a fully connected layer is followed by a dropout layer
    dropout_rate: float, defines the dropout rate for the dropout layers
    """

    # DATA HANDLING
    # Define additional parameters for the neural network
    act_func, act_title, batch_size, class_number, feature_number, layers, dropout, fc_layer_units, test_labels, \
        test_vec, training_labels, training_vec = define_parameters(act_func, dropout, fc_layer_units, labels, vec)

    # Set the start time for running the neural network
    start_time_network = time.time()

    # GRAPH DEFINITION
    # Define the neural network architecture
    cost, data_input, label_input, minimization_step, pred, softmaxed_pred = ANN_architecture(feature_number, class_number,
                                                                              fc_layer_units, dropout, dropout_rate,
                                                                              act_func, learning_rate)

    # Run the neural network
    sess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=8))
    sess.run(tf.global_variables_initializer())
    working_dir = os.path.dirname(os.path.abspath(__file__))
    writer = tf.summary.FileWriter(working_dir + "/graphs/tensorboard", graph=sess.graph)

    # CHECKS
    # Train the neural network
    all_accuracy, all_costs, all_test_accuracy, all_test_costs, test_labels, test_vec, training_labels, training_vec = \
        network_training(batch_size, cost, iterations, data_input, label_input, minimization_step, softmaxed_pred, sess,
                         test_labels, test_vec, training_labels, training_vec)

    # Get the end accuracy of the training, test, and validation set
    end_training_accuracy = get_accuracy(data_input, label_input, softmaxed_pred, sess, training_labels, training_vec)
    end_test_accuracy = get_accuracy(data_input, label_input, softmaxed_pred, sess, test_labels, test_vec)
    end_val_accuracy = get_accuracy(data_input, label_input, softmaxed_pred, sess, val_labels, val_vec)
    print("\ntraining accuracy: {0:.2f}%".format(end_training_accuracy))
    print("test accuracy: {0:.2f}%".format(end_test_accuracy))
    print("validation accuracy: {0:.2f}%".format(end_val_accuracy))
    # Get the ROC AUC scores of the training, test and validation set
    roc_training = get_ROC_AUC(data_input, label_input, pred, sess, training_labels, training_vec)
    roc_test = get_ROC_AUC(data_input, label_input, pred, sess, test_labels, test_vec)
    roc_val = get_ROC_AUC(data_input, label_input, pred, sess, val_labels, val_vec)
    print("\nROC-AUC training: {:.2f}".format(roc_training))
    print("ROC-AUC test: {:.2f}".format(roc_test))
    print("ROC-AUC validation: {:.2f}".format(roc_val))
    # Get some statistics about the cost function
    start_test_cost = all_test_costs[0]
    end_test_cost = all_test_costs[-1]
    min_test_cost = min(all_test_costs)
    min_test_cost_idx = all_test_costs.index(min_test_cost)
    print("\noptimum number of iterations: {}".format(min_test_cost_idx))

    # Get the weights after training
    influence_weights(sess, n_neighbours, data_size)

    # Close the TensorFlow session
    writer.close()
    sess.close()

    # Get the total running time of the neural network
    network_run_time = time.time() - start_time_network

    # GRAPHS
    # Store the statistical results
    store_statistics(all_features, data_size, n_neighbours, layers, act_title, dropout_rate, learning_rate, iterations,
                     min_test_cost_idx, min_test_cost, start_test_cost, end_test_cost, end_training_accuracy,
                     end_test_accuracy, end_val_accuracy, roc_training, roc_test, roc_val, network_run_time)

    # Create a plot of the cost function
    error_plot(all_costs, all_test_costs, learning_rate, end_training_accuracy, end_test_accuracy, end_val_accuracy,
               layers, data_size, n_neighbours, dropout_rate)

    # Create a plot of the accuracy
    accuracy_plot(all_accuracy, all_test_accuracy, layers, data_size, n_neighbours, learning_rate, dropout_rate)


def define_parameters(act_func, dropout, fc_layer_units, labels, samples):
    """ Convert some parameters in the right format

    act_funct: string, defines which activation function will  be used for training the neural network
    dropout: list, contains boolean values that defines if a fully connected layer is followed by a dropout layer
    fc_layers_units: list, contains integer values that defines the number of nodes for the fully connected layers
    labels: one hot encoded numpy array, contains data labels
    samples: numpy array, contains features scores with shape (samples, number of features * number of neighbouring
     positions)
    """



    # Define a training and test set
    test_size = 0.1  # training is set on 90%
    training_vec, test_vec, training_labels, test_labels = train_test_split(samples, labels, test_size=test_size)

    # Get the batch size
    batch_percentage = 0.1  # There is chosen to use a batch size of 10%
    batch_size = int(training_vec.shape[0] * batch_percentage)

    # Get the number of features
    feature_number = training_vec.shape[1]

    # Get the number of classes
    class_number = len(np.unique(labels))

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
    layers = []
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
    elif act_func == 'tanh' or act_func == 'tan' or act_func == 't':
        act_func = tf.tanh
        act_title = 'tanH'
    else:
        act_func = None
        act_title = 'none'

    return act_func, act_title, batch_size, class_number, feature_number, layers, dropout_booleans, int_layer_units, \
           test_labels, test_vec, training_labels, training_vec


def ANN_architecture(feature_number, class_number, layer_units, dropout, dropout_rate, act_func, learning_rate):
    """ Defines the neural network architecture with name scoping for visualization in TensorBoard

    feature_number: integer, defines the number of considered features
    class_number: integer, defines the number of class labels
    layer_units: list, contains the number of nodes for every fully connected layer
    dropout: list, contains boolean values which refers if a dropout layer occurs after a fully connected layer
    dropout_rate: float, defines the dropout rate for the dropout layers
    act_func: Tensor, defines the activation function that will used in the fully connected layers
    learning_rate: float, defines the learning rate for training the network
    """

    # Define the input layer
    with tf.variable_scope("input"):
        # Define a placeholder for both the data and labels
        data_input = tf.placeholder(tf.float32, shape=[None, feature_number], name="data_input")
        label_input = tf.placeholder(tf.float32, shape=[None, class_number], name="label_input")

    # Define the fully connected and dropout layers
    with tf.variable_scope("layers"):
        # The 'index_offset indicates which layer number it is; the offset increase if dropout layers are included
        index_offset = 0
        # The 'previous_dropout' indicates if the previous layer is a dropout layer or not
        previous_dropout = False
        # Loop through the defined hidden layers
        for index, n_units in enumerate(layer_units):
            # The first layer uses the data as input
            if index == 0:
                name_layer = 'fc-layer{}'.format(index)
                fully_connected_layer = tf.layers.dense(inputs=data_input, units=layer_units[index],
                                                        activation=act_func, use_bias=True, name=name_layer)
                # Check if a drop out layer is used after the fully connected layer
                if dropout[index]:
                    previous_dropout = True
                    index_offset += 1
                    name_layer = 'dr-layer{}'.format(index + index_offset)
                    drop_out_layer = tf.layers.dropout(inputs=fully_connected_layer, rate=dropout_rate, name=name_layer)
                else:
                    previous_dropout = False
            # Other layers use the previous fully connected or drop out layer as input
            else:
                # Check if the previous layer is a dropout layer
                if previous_dropout:
                    name_layer = 'fc-layer{}'.format(index + index_offset)
                    fully_connected_layer = tf.layers.dense(inputs=drop_out_layer, units=layer_units[index],
                                                            activation=act_func, use_bias=True, name=name_layer)
                # Check if the previous layer is a fully connected layer
                else:
                    name_layer = 'fc-layer{}'.format(index + index_offset)
                    fully_connected_layer = tf.layers.dense(inputs=fully_connected_layer, units=layer_units[index],
                                                            activation=act_func, use_bias=True, name=name_layer)
                # Check if a drop out layer is used after the fully connected layer
                if dropout[index]:
                    previous_dropout = True
                    index_offset += 1
                    name_layer = 'dr-layer{}'.format(index + index_offset)
                    drop_out_layer = tf.layers.dropout(inputs=fully_connected_layer, rate=dropout_rate, name=name_layer)
                else:
                    previous_dropout = False

    # Define the output layer
    with tf.variable_scope('output'):
        # Check if the last layer is a dropout or fully connected layer
        if len(dropout) == 1:
            last_dropout = dropout[0]
        else:
            last_dropout = dropout[-1]

        # Define the output layer based on the previous fully connected or dropout layer
        if last_dropout:
            # Activation function is not needed for the last layer, because the cross entropy already use softmax
            pred = tf.layers.dense(inputs=drop_out_layer, units=class_number, activation=None, use_bias=False,
                                   name="prediction")
        else:
            pred = tf.layers.dense(inputs=fully_connected_layer, units=class_number, activation=None,
                                   use_bias=True, name="prediction")
        # Use float64 to prevent under/overflow
        pred = tf.cast(pred, tf.float64)
        softmaxed_pred = tf.nn.softmax(pred, name="SoftmaxedPrediction")

    # Define the loss function
    with tf.variable_scope("Cost"):
        # Normalize the cost for the batch size (denominator) with euclidean distance
        # cost = tf.nn.l2_loss(pred - label_input)\
        # If the output layer contains 1 class the sigmoid cross entropy can be used
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label_input)
        cost = tf.reduce_mean(cost) #* 100

    # Define how to optimize the neural network
    with tf.variable_scope("optimizer"):
        # minimization_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        minimization_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    return cost, data_input, label_input, minimization_step, pred, softmaxed_pred


def network_training(batch_size, cost, iterations, data_input, label_input, minimization_step, pred, sess,
                     test_labels, test_vec, training_labels, training_vec):
    """ Function that trains the neural network

    batch_size: integer, defines the batch size for training
    cost: Tensor, containing float values to minimize
    iterations: integer, defines the number of iteration for training the neural network
    data_input: TensorFlow placeholder, contains a defined subpart of the data samples
    label_input: TensorFlow placeholder, contains a defined subpart of the data labels
    pred: Tensor, contains predicted class labels
    sess: TensorFlow object, encapsulates the environment in which operation objects are executed and Tensor objects
     are evaluated
    test_labels: one hot encoded numpy array, contains data labels of the test set
    test_vec: numpy array, contains features scores with shape (samples, number of features * number of neighbouring
     positions) of the test set
    training_labels: one hot encoded numpy array, contains data labels of the training set
    training_vec: numpy array, contains features scores with shape (samples, number of features * number of neighbouring
     positions) of the training set
    """

    # Define list to keep track of the loss and accuracy
    all_costs = []
    all_test_costs = []
    all_accuracy = [0]
    all_test_accuracy = [0]

    # Set the range for looping to train the neural network
    for i in range(iterations + 1):
        # To prevent slicing will be out of range
        offset = int(i * batch_size % len(training_vec))

        # Epoch wise training data shuffling
        if offset + batch_size >= len(training_vec):
            training_vec, training_labels = shuffled_copies(training_vec, training_labels)
            test_vec, test_labels = shuffled_copies(test_vec, test_labels)

        batch_samples = training_vec[offset:offset + batch_size]
        batch_labels = training_labels[offset:offset + batch_size]

        # Calculate for every iteration the training and test loss
        _, current_cost = sess.run([minimization_step, cost],
                                   # feed_dict={data_input: training_vec,
                                   #            label_input: training_labels})
                                   feed_dict={data_input: batch_samples,
                                              label_input: batch_labels})

        test_cost = sess.run(cost, feed_dict={data_input: test_vec,
                                              label_input: test_labels})

        # Get the accuracy for the first 100 iterations in step of 5, because this is a steep part in the graph
        if i in range(0, 101) and i % 5 == 0:
            accuracy_sess = get_accuracy(data_input, label_input, pred, sess, training_labels, training_vec,
                                         offset=offset, batch_size=batch_size)
            accuracy_sess_test = get_accuracy(data_input, label_input, pred, sess, test_labels, test_vec)

        # Thereafter get the accuracy for every 20th iteration
        if i in range(102, iterations + 1) and i % 20 == 0:
            accuracy_sess = get_accuracy(data_input, label_input, pred, sess, training_labels, training_vec,
                                         offset=offset, batch_size=batch_size)
            accuracy_sess_test = get_accuracy(data_input, label_input, pred, sess, test_labels, test_vec)

        # Check every 50th loop how well the prediction is
        if i % 50 == 0:
            print("accuracy  : {0:.2f}%".format(accuracy_sess))
            print("STEP: {} | Cost: {}".format(i, current_cost))

        # Add the los and accuracy to the defined lists
        all_costs.append(current_cost)
        all_test_costs.append(test_cost)
        all_accuracy.append(accuracy_sess)
        all_test_accuracy.append(accuracy_sess_test)

    return all_accuracy, all_costs, all_test_accuracy, all_test_costs, test_labels, test_vec, training_labels, \
           training_vec


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


def get_accuracy(data_input, label_input, pred, sess, labels, samples, offset=None, batch_size=None):
    """ Get the accuracy in percentage while training the neural network

    data_input: TensorFlow placeholder, contains a defined subpart of the data samples
    label_input: TensorFlow placeholder, contains a defined subpart of the data labels
    pred: Tensor, contains predicted class labels
    sess: TensorFlow object, encapsulates the environment in which operation objects are executed and Tensor objects
     are evaluated
    labels: one hot encoded numpy array, contains data labels
    samples: numpy array, contains features scores with shape (samples, number of features * number of neighbouring
     positions)
    """

    # Get the predicted labels and the real labels
    if offset is None and batch_size is None:
        pred_out, label_input_out = sess.run([pred, label_input],
                                             feed_dict={data_input: samples, label_input: labels})
    else:
        batch_samples = samples[offset:offset + batch_size]
        batch_labels = labels[offset:offset + batch_size]
        pred_out, label_input_out = sess.run([pred, label_input], feed_dict={data_input: batch_samples,
                                                                             label_input: batch_labels})

    # Compare the true labels with the predicted labels
    correct_prediction = tf.equal(tf.argmax(pred_out, axis=1), tf.argmax(label_input_out, axis=1))
    # Check the accuracy by counting the miss-classifications
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_sess = sess.run(accuracy, feed_dict={label_input: labels})

    # Get the accuracy in percentage
    perc_accuracy = accuracy_sess * 100

    return perc_accuracy


def get_ROC_AUC(data_input, label_input, pred, sess, labels, samples):
    """ Get the ROC AUC accuracy in percentage while training the neural network

    data_input: TensorFlow placeholder, contains a defined subpart of the data samples
    label_input: TensorFlow placeholder, contains a defined subpart of the data labels
    pred: tensor, contains predicted class labels
    sess: TensorFlow object, encapsulates the environment in which operation objects are executed and Tensor objects
     are evaluated
    labels: one hot encoded numpy array, contains data labels
    samples: numpy array, contains features scores with shape (samples, number of features * number of neighbouring
     positions)
    """

    # FUNCTION DOES NOT WORK PROPERLY, ROC AUC IS NOW ALWAYS 50%
    #TODO: fix ROC AUC

    #  Get the predicted and the real labels
    pred, labels = sess.run([pred, label_input], feed_dict={data_input: samples, label_input: labels})

    # ROC AUC can not handle negative values, so they should be converted
    # The softmax function is used to get only positive values
    #TODO: this is where the bug is coming from
    sum_exp_pred = np.sum(np.exp(pred))
    pred = (np.exp(pred)) / sum_exp_pred

    # Get the ROC scores
    _, roc = tf.metrics.auc(labels=labels, predictions=pred)

    # Run the local variables to get the ROC AUC accuracy
    sess.run(tf.local_variables_initializer())

    # Run the session to obtain the accuracy
    roc_auc = sess.run(roc*100)

    return roc_auc


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
    layer_number = 0
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


def store_statistics(all_features, n_samples, n_neighbours, layers, act_func, dropout_rate, learning_rate, iterations,
                     optimum_idx, optimum_loss, start_test_cost, end_test_cost, accuracy_train, accuracy_test,
                     accuracy_val, ROC_AUC_train, ROC_AUC_test, ROC_AUC_val, run_time):
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

    # Get the variables in the right format
    all_features = all_features.lower()
    if all_features == 'y' or all_features == 'yes':
        all_features = "all conservation-based scores"
    else:
        all_features = "only PhastCon primate scores"

    if n_neighbours == 0:
        title = "excluding"
    else:
        title = "including"

    # Get the file name where the statistical results will be written to
    working_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = working_dir + "/output_ANN/stats_ANN.txt"

    # Extend the file with the results in the corresponding data types
    with open(file_name, 'a') as output:
        output.write("\n{:s}\t{:d}\t{:s}\t{:d}\t{:s}\t{:s}\t{:.2f}\t{:f}\t{:d}\t{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t"
                     "{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.0f}".format(
                     all_features, n_samples, title, n_neighbours, layers, act_func, dropout_rate, learning_rate,
                     iterations, optimum_idx, optimum_loss, start_test_cost, end_test_cost, accuracy_train,
                     accuracy_test, accuracy_val, ROC_AUC_train, ROC_AUC_test, ROC_AUC_val, run_time))

    output.close()


def error_plot(training_costs, test_costs, learning_rate, accuracy, test_accuracy, val_accuracy, layers, data_size,
               n_neighbours, dropout_rate):
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
    dropout_rate: float, defines the dropout rate for the dropout layers
    """

    plt.plot(training_costs, label="Training loss")
    plt.plot(test_costs, label="Test loss")
    plt.xlabel("Iterations", size='medium')
    plt.ylabel("Cost function (%)", size='medium')
    plt.suptitle("Cost function while training the neural network", size='medium', ha='center')
    plt.title("layers: {} with dropout rate of {}, learning rate: {}".format(layers, dropout_rate, learning_rate),
              size='small', ha='center')
    plt.figtext(0.77, 0.35, "Training accuracy\n{0:.2f}%".format(accuracy), size='medium')
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
    saving(working_dir + "/output_ANN/error_plots/{}_error_{}".format(n_neighbours, data_size))


def accuracy_plot(training, test, layers, data_size, n_neighbours, learning_rate, dropout_rate):
    """ Return plot of the accuracy during the training of the neural network

    training: list, contains training accuracy while training the neural network
    test: list, contains test accuracy while training the neural network
    layers: string, defines which layers are used with how many hidden nodes
    data_size: integer, defines the number of samples in the data
    n_neighbours: integer, indicates how many neighbours are included
    learning_rate: float, strictness of learning while training the neural network
    dropout_rate: float, defines the dropout rate for the dropout layers
    """

    plt.figure()
    plt.plot(training, label="Training")
    plt.plot(test, label="Test")
    plt.xlabel("Iterations", size='medium')
    plt.ylabel("Accuracy function (%)", size='medium')
    plt.suptitle("Accuracy function while training the neural network", size='medium', ha='center')
    plt.title("layers: {} with dropout rate of {}, learning rate: {}".format(layers, dropout_rate, learning_rate),
              size='small', ha='center')
    if n_neighbours == 0:
        plt.figtext(0.83, 0.80, "Neighbours\nexcluded", size='medium')
    else:
        plt.figtext(0.83, 0.80, "{} neighbours\nincluded".format(n_neighbours), size='medium')
    plt.figtext(0.83, 0.70, "{}\nsamples".format(data_size), size='medium')
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

    # # Specify the variable if you do not run from the command line
    # data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/5_200000.h5"
    # val_data_directory = "/mnt/scratch/kersj001/data/output/normalized_data/5_26172.h5"
    # all = 'yes'
    # snp_neighbour = 'neighbour'
    # # snp_neighbour = 'snp'
    # learning_rate = 0.1
    # dropout_rate = 0.5
    # activation_function = 'r'  # tf.nn.relu
    # iterations = 1000
    # fc_nodes = '5,7'
    # dropout_layer = 'F,F'

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
    # Specify the nodes for the fully connected layers
    parser.add_option("-n", "--nodes", dest="fc_nodes", help="List that contains integer values that defines the \
        number of nodes for the fully connected layer", default="4")
    # Specify if dropout layers occurs after a fully connected layer
    parser.add_option("-b", "--dropout", dest="dropout_layers", help="List that contains boolean values that defines \
        if a fully connected layer is followed by a dropout layer", default="False")

    # Get the command line options for reading the data for both the benign and deleterious SNPs
    (options, args) = parser.parse_args()
    data_directory = options.data
    val_data_directory = options.validation_data
    all = options.all_features
    snp_neighbour = options.snp_neighbour
    learning_rate = float(options.learning_rate)
    dropout_rate = float(options.dropout_rate)
    activation_function = options.activation
    iterations = int(options.iterations)
    fc_nodes = options.fc_nodes
    dropout_layer = options.dropout_layers

    return data_directory, val_data_directory, all, snp_neighbour, learning_rate, dropout_rate, activation_function, \
           iterations, fc_nodes, dropout_layer


if __name__ == "__main__":
    # Keep track of the running time
    start_time = time.time()

    # Get the given arguments including the hyperparameters for the neural network
    data_directory, val_data_directory, all_features, snp_neighbour, learning_rate, dropout_rate, activation_function, \
        iterations, fc_nodes, dropout_layer = get_arguments()

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
    neural_network(samples, labels, samples_val, labels_val, data_size, n_neighbours, learning_rate, all_features,
                   iterations, activation_function, fc_nodes, dropout_layer, dropout_rate)

    print("\n----- running time: {} seconds -----".format(round(time.time() - start_time), 2))
