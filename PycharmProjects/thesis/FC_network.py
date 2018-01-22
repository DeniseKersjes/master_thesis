import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from combine_features import inputCNN


def neural_network(vec, labels, title='neighbour'):
    """ Return predicted labels and actual labels with error rate and accuracy after training the fully connected
    neural network

    vec: numpy array, contains features
    labels: one hot encoded array, contains data labels
    title: string, indicate if the neural network is running with only the features of the SNP of interest or also
        includes the features of the neighbouring positions, default=neighbour
    """

    # HYPER PARAMETERS
    # There is chosen to use a batch size of 10%
    batch_size = int(vec.shape[0] * 0.1)
    act_func = tf.nn.relu
    pred_act_func = tf.nn.softmax
    layer_1_units = 30
    layer_2_units = 15
    number_hidden_layers = 2
    learning_rate = 0.01
    iterations = 500

    # DATA HANDLING
    # Define training and test set
    test_size = 0.2  # training is set on 80%
    training_vec, test_vec, training_labels, test_labels = train_test_split(vec, labels, test_size=test_size)
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
        fully_connected_2 = tf.layers.dense(inputs=fully_connected_1, units=layer_2_units, activation=act_func,
                                            use_bias=True, name="fc-layer2")

    with tf.variable_scope('output'):
        pred = tf.layers.dense(inputs=fully_connected_2, units=2, activation=pred_act_func, use_bias=True,
                               name="prediction")

    with tf.variable_scope("prediction"):
        # Normalize the cost for the batch size (denominator) with euclidean distance
        cost = tf.nn.l2_loss(pred - label_input)

    with tf.variable_scope("optimizer"):
        minimization_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./graphs/tensorboard", graph=sess.graph)

    # CHECKS
    # Set the range for looping to train the neural network
    all_costs = []
    for i in range(iterations):
        # To prevent slicing will be out of range
        offset = int(i * batch_size % len(training_vec))
        # With feed_dict the placeholder is filled in
        _, current_cost = sess.run([minimization_step, cost],
                                   feed_dict={data_input: training_vec[offset:offset + batch_size],
                                              label_input: training_labels[offset:offset + batch_size]})
        all_costs += [current_cost]
        # Check every 10th loop what the cost is
        if i % 5 == 0:
            print("STEP: {} | Average cost: {}".format(i, np.mean(current_cost)))
        # Check every 50th loop how well the prediction is
        if i % 50 == 0:
            pred_out, label_input_out = sess.run([pred, label_input],
                                                 feed_dict={data_input: training_vec[offset:offset + batch_size],
                                                            label_input: training_labels[offset:offset + batch_size]})
            # Compare the true labels with the predicted labels
            print("\nlabels     : ", np.argmax(label_input_out, axis=1), )
            print("predictions: ", np.argmax(pred_out, axis=1))
            correct_prediction = tf.equal(tf.argmax(pred_out, axis=1), tf.argmax(label_input_out, axis=1))

            # Check the accuracy by counting the miss-classifications
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_sess = sess.run(accuracy, feed_dict={label_input: test_labels})
            print("accuracy   :  {0:.2f}".format(accuracy_sess * 100), '%\n')

    test_cost = sess.run(cost, feed_dict={data_input: test_vec,
                                          label_input: test_labels})

    print("\ntest_cost: {} ".format(test_cost))
    print("end accuracy of the predictions: {0:.2f}".format(accuracy_sess * 100), '%')

    predictions, _ = sess.run([pred, label_input], feed_dict={data_input: training_vec,
                                                              label_input: training_labels})

    writer.close()
    sess.close()

    error_plot(all_costs, number_hidden_layers, learning_rate, title)

    return all_costs


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


def error_plot(costs, n_hidden_layers, learning_rate, title):
    """ Return plot of the decreasing error during the training of the neural network

    costs: list, contain error costs
    learning_rate: float, strictness of learning while training the neural network
    n_hidden_layers: integer, number of hidden layers
    """

    plt.plot(costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost function")
    plt.title("Cost function while training the neural network \n{} hidden layers, learning rate: {}".format(
        n_hidden_layers, learning_rate))
    saving("./output_ANN/error_{}".format(title))


def saving(file_path):
    """ Saves the plot in the correct directory with an unique name

    file_path: string, path direction to where the plot will be saved
    """

    index_saving = 1
    while os.path.exists(file_path+"_{}.png".format(index_saving)):
        index_saving += 1
    plt.savefig(file_path+"_{}.png".format(index_saving))


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


if __name__ == "__main__":

    directory_ben = "/mnt/scratch/kersj001/data/output/test/test_ben/"
    ben_data = inputCNN(directory_ben).combine(directory_ben)
    directory_del = "/mnt/scratch/kersj001/data/output/test/test_del/"
    del_data = inputCNN(directory_del).combine(directory_del)

    # Run the neural network with the SNP of interest and its neighbouring positions
    # samples_neighbour, labels_neighbour = data_parser(ben_data, del_data, snp_or_neighbour='neighbour')
    # neural_network(samples_neighbour, labels_neighbour, title='neighbour')

    # Run the neural network with only the SNP of interest
    samples_neighbour, labels_neighbour = data_parser(ben_data, del_data, snp_or_neighbour='SNP')
    neural_network(samples_neighbour, labels_neighbour, title='SNP')
