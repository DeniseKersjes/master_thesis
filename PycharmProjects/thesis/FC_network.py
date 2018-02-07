import time
import h5py
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
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


def neural_network(vec, labels, data_size, learning_rate=0.01, title='neighbour'):
    """ Return predicted labels and actual labels with error rate and accuracy after training the fully connected
    neural network

    vec: numpy array, contains features
    labels: one hot encoded array, contains data labels
    title: string, indicate if the neural network is running with only the features of the SNP of interest or also
        includes the features of the neighbouring positions, default=neighbour
    """

    # HYPER PARAMETERS
    act_func = tf.nn.relu
    pred_act_func = tf.nn.softmax
    layer_1_units = 5
    # layer_2_units = 7
    # layer_3_units = 5
    number_hidden_layers = 1
    learning_rate = learning_rate
    iterations = 500

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
    with tf.variable_scope("fully-connected"):
        fully_connected_1 = tf.layers.dense(inputs=data_input, units=layer_1_units, activation=act_func, use_bias=True,
                                            name="fc-layer1")
        # fully_connected_2 = tf.layers.dense(inputs=fully_connected_1, units=layer_2_units, activation=act_func,
        #                                     use_bias=True, name="fc-layer2")
        # fully_connected_3 = tf.layers.dense(inputs=fully_connected_2, units=layer_3_units, activation=act_func,
        #                                     use_bias=True, name="fc-layer3")

    with tf.variable_scope('output'):
        pred = tf.layers.dense(inputs=fully_connected_1, units=2, activation=pred_act_func, use_bias=True,
                               name="prediction")

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
    all_accuracy = []
    for i in range(iterations+1):
        # To prevent slicing will be out of range
        offset = int(i * batch_size % len(training_vec))
        # Epoch wise training data shuffling
        if offset + batch_size >= len(training_vec):
            training_vec, training_labels = shuffled_copies(training_vec, training_labels)

        # With feed_dict the placeholder is filled in

        # Check every 50th loop how well the prediction is
        if i % 100 == 0:
            pred_out, label_input_out = sess.run([pred, label_input],
                                                 feed_dict={data_input: training_vec[offset:offset + batch_size],
                                                            label_input: training_labels[offset:offset + batch_size]})
            # Compare the true labels with the predicted labels
            correct_prediction = tf.equal(tf.argmax(pred_out, axis=1), tf.argmax(label_input_out, axis=1))

            # Check the accuracy by counting the miss-classifications
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_sess = sess.run(accuracy, feed_dict={label_input: test_labels})
            print("accuracy   :  {0:.2f}".format(accuracy_sess * 100), '%\n')
            all_accuracy.append(accuracy_sess)

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
        all_accuracy.append(accuracy_sess)

    predictions, test_cost = sess.run([pred, cost], feed_dict={data_input: test_vec,
                                                               label_input: test_labels})

    print("\ntest_cost: {} ".format(np.mean(test_cost)))
    print("end accuracy of the predictions: {0:.2f}".format(accuracy_sess * 100), '%')

    min_test_cost = min(all_test_costs)
    print("optimum number of iterations: ", all_test_costs.index(min_test_cost))

    writer.close()
    sess.close()

    error_plot(all_costs, all_test_costs, number_hidden_layers, learning_rate, accuracy_sess, data_size, title,
               y_axis="Cost function")

    return all_costs


def error_plot(training_costs, test_costs, n_hidden_layers, learning_rate, accuracy, data_size, title, y_axis):
    """ Return plot of the decreasing error during the training of the neural network

    costs: list, contain error costs
    learning_rate: float, strictness of learning while training the neural network
    n_hidden_layers: integer, number of hidden layers
    """

    plt.figure()
    plt.plot(training_costs, label="Training")
    plt.plot(test_costs, label="Test")
    plt.xlabel("Iterations")
    plt.ylabel(y_axis)
    plt.title("Cost function while training the neural network \n{} hidden layer(s), learning rate: {}".format(
        n_hidden_layers, learning_rate))
    plt.figtext(0.83, 0.35, "End accuracy\n{0:.2f}%".format(accuracy * 100))
    if title == "snp" or title == "SNP":
        plt.figtext(0.83, 0.80, "Neighbours\nexcluded")
    else:
        plt.figtext(0.83, 0.80, "Neighbours\nincluded")
    plt.legend(loc='right', bbox_to_anchor=(1.30, 0.5))
    plt.subplots_adjust(right=0.8)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    saving(working_dir + "/output_ANN/error_{}_{}".format(title, data_size))


def saving(file_path):
    """ Saves the plot in the correct directory with an unique name

    file_path: string, path direction to where the plot will be saved
    """

    index_saving = 1
    while os.path.exists(file_path + "_{}.png".format(index_saving)):
        index_saving += 1
    plt.savefig(file_path + "_{}.png".format(index_saving))


def data_reading(data_directory):
    """ Returns the converted H5py data set into numpy array

    data_directory: string, defined path ending with the desired file name (.h5)
    """

    # Read the data
    h5f = h5py.File(data_directory, 'r')

    # The data set name is the name of the path where the data file can be found
    data = h5f[data_directory.split('/')[-2]][:]

    # Close the H5py file
    h5f.close()

    return data


def data_parser(data_ben, data_del, snp_or_neighbour='neighbour'):
    """ Return the labels and the features as two separated numpy arrays

    data_ben: numpy array with shape (number of samples, number of features, number of nucleotides)
    data_del: numpy array with shape (number of samples, number of features, number of nucleotides)
    snp_or_neighbour: string, indicates if the neural network will run with only the features of the SNP of interest
        or also includes the features of the neighbouring positions, default=neighbour
    """

    # Reshape the data into a 2D array
    if snp_or_neighbour == "SNP" or snp_or_neighbour == "snp":
        # Get only the features of the SNP of interest, which is located at the middle position of the data
        index_SNPi_ben = (data_ben.shape[2] - 1) / 2  # -1 for the SNP of interest
        index_SNPi_del = (data_del.shape[2] - 1) / 2
        data_ben = data_ben[:, :, int(index_SNPi_ben)]
        data_del = data_del[:, :, int(index_SNPi_del)]
    else:
        # Reshape the data into a 2D array including the neighbouring positions
        data_ben = data_ben.reshape(data_ben.shape[0], -1)
        data_del = data_del.reshape(data_del.shape[0], -1)

    # Replace NaN values with 0
    data_ben = np.nan_to_num(data_ben)
    data_del = np.nan_to_num(data_del)

    # Combine the benign and deleterious SNPs to 1 array
    samples = np.concatenate((data_ben, data_del), axis=0)  # 0 to put the arrays behind each other

    # Get the corresponding labels; the SNPs that are benign have class label 0, and the deleterious SNPs class label 1
    labels_ben = [0] * data_ben.shape[0]
    label_del = [1] * data_del.shape[0]
    # Combine the benign and deleterious labels to 1 numpy array
    labels = np.array(labels_ben + label_del)

    # Convert the data into one hot encoded data
    labels = initialization_based(labels)

    return samples, labels


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


if __name__ == "__main__":
    # Keep track of the running time
    start_time = time.time()

    # # If you do not run from the command line
    # # Read the data for both the deleterious and benign SNPs
    # # Path directory for the benign SNPs always ending with combined_ben.h5
    # # data_directory_ben = "/mnt/scratch/kersj001/data/output/10_thousand_ben/combined_ben.h5"
    # data_directory_ben = "/mnt/scratch/kersj001/data/output/1_thousand_ben/combined_ben.h5"
    # #
    # # Path directory for the deleterious SNPs always ending with combined_del.h5
    # # data_directory_del = "/mnt/scratch/kersj001/data/output/10_thousand_del/combined_del.h5"
    # data_directory_del = "/mnt/scratch/kersj001/data/output/1_thousand_del/combined_del.h5"
    #
    # learning_rate = 0.01
    # snp_neighbour = 'neighbour'
    # # snp_neighbour = 'snp'
    # data_size = 2000

    # Run the neural network with only the SNP of interest
    print("\n\nEXCLUDES NEIGHBOURING POSITIONS\n")
    samples_neighbour, labels_neighbour = data_parser(data_ben, data_del, snp_or_neighbour='SNP')
    neural_network(samples_neighbour, labels_neighbour, title='SNP')

    # Run the neural network with the SNP of interest and its neighbouring positions
    print("\n\nINCLUDES NEIGHBOURING POSITIONS\n")
    samples_neighbour, labels_neighbour = data_parser(data_ben, data_del, snp_or_neighbour='neighbour')
    neural_network(samples_neighbour, labels_neighbour, title='neighbour')

    # Specify the options for running from the command line
    parser = OptionParser()

    # Specify the data directory for the benign and deleterious SNPs
    parser.add_option("-b", "--ben", dest="benign", help="Path to the output of the 'combine_features.py' script that \
            generates a H5py file with the compressed numpy array containing feature scores of benign SNPs and its \
            neighbouring features", default="")
    parser.add_option("-d", "--del", dest="deleterious", help="Path to the output of the 'combine_features.py' script \
            that generates a H5py file with the compressed numpy array containing feature scores of deleterious SNPs \
            and its neighbouring features", default="")
    parser.add_option("-l", "--learning", dest="learning_rate", help="Float that defines the learning rate of for \
            training the neural network", default=0.01)
    parser.add_option("-n", "--datasize", dest="data_size", help="Integer that defines the number of samples in the \
            data set", default=2000)
    parser.add_option("-s", "--snp", dest="snp_neighbour", help="String that indicates if the surrounding neighbours \
            will be included ('n') or excluded ('s')", default="n")

    # Get the command line options for reading the data for both the benign and deleterious SNPs
    (options, args) = parser.parse_args()
    data_directory_ben = options.benign
    data_directory_del = options.deleterious
    learning_rate = float(options.learning_rate)
    data_size = options.data_size
    snp_neighbour = options.snp_neighbour

    data_ben = data_reading(data_directory_ben)
    data_del = data_reading(data_directory_del)

    samples_neighbour, labels_neighbour = data_parser(data_ben, data_del, snp_or_neighbour=snp_neighbour)
    neural_network(samples_neighbour, labels_neighbour, data_size, learning_rate=learning_rate, title=snp_neighbour)

    print("\n----- running time: {} seconds -----".format(round(time.time() - start_time), 2))
