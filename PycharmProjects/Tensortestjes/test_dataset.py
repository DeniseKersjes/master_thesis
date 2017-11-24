import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def neural_network(vec, labels):
    """ Return predicted labels and actual labels with error rate and accuracy after training the fully connected
    neural network

    vec: numpy array, contains features
    labels: one hot encoded array, contains data labels
    """

    # HYPER PARAMETERS
    # there is chosen to use a batch size of 10% (data contain 100 samples)
    batch_size = 10
    act_func = tf.nn.relu
    pred_act_func = tf.nn.softmax
    layer_1_units = 10
    learning_rate = 0.01

    # DATA HANDLING
    # define training and test set, training is set on 80%
    training_vec, test_vec, training_labels, test_labels = train_test_split(vec, labels, test_size=0.2)
    feature_number = 2  # == training_vec.shape[1]
    class_number = 2  # == len( np.unique(labels) )

    # GRAPH DEFINITION
    # use scope to get a nicely lay-out in TensorBoard
    with tf.variable_scope("input"):
        # define a placeholder for both the data and labels
        data_input = tf.placeholder(tf.float32, shape=[None, feature_number], name="data_input")
        label_input = tf.placeholder(tf.float32, shape=[None, class_number], name="label_input")

    # Fully-connected layers
    with tf.variable_scope("fully-connected"):
        fully_connected_1 = tf.layers.dense(inputs=data_input, units=layer_1_units, activation=act_func, use_bias=True,
                                            name="fc-layer")

    with tf.variable_scope('output'):
        pred = tf.layers.dense(inputs=fully_connected_1, units=2, activation=pred_act_func, use_bias=True,
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
    # set the range for looping to train the neural network
    for i in range(250):
        # to prevent slicing will be out of range
        offset = int(i * batch_size % len(training_vec))
        # with feed_dict the placeholder is filled in
        _, current_cost = sess.run([minimization_step, cost],
                                   feed_dict={data_input: training_vec[offset:offset + batch_size],
                                              label_input: training_labels[offset:offset + batch_size]})
        # check every 10th loop what the cost is
        if i % 10 == 0:
            print("STEP: {} | Average cost: {}".format(i, np.mean(current_cost)))
        # check every 50th loop how well the prediction is
        if i % 50 == 0:
            pred_out, label_input_out = sess.run([pred, label_input],
                                                 feed_dict={data_input: training_vec[offset:offset + batch_size],
                                                            label_input: training_labels[offset:offset + batch_size]})
            # compare the true labels with the predicted labels
            print("\nlabels     : ", np.argmax(label_input_out, axis=1), )
            print("predictions: ", np.argmax(pred_out, axis=1))
            correct_prediction = tf.equal(tf.argmax(pred_out, axis=1), tf.argmax(label_input_out, axis=1))

            # check the accuracy by counting the miss-classifications
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_sess = sess.run(accuracy, feed_dict={label_input: test_labels})
            print("accuracy   :  {}".format(accuracy_sess*100), '%\n')

    test_cost = sess.run(cost, feed_dict={data_input: test_vec,
                                          label_input: test_labels})
    print("\ntest_cost: {} ".format(test_cost))
    print("end accuracy of the predictions: {}".format(accuracy_sess*100), '%')

    writer.close()
    sess.close()
    return 'Working!'


def plot(vec):
    """ Return contour plot of features

    vec: numpy array, contains features
    """

    # ADDITIONAL CODE, to see what the ranges are of the features
    # get a list containing all the first features and second features, respectively
    # feature_1 = list(map(lambda x: x[0], vec))
    # feature_2 = list(map(lambda x: x[1], vec))
    # x = np.arange(min(feature_1), max(feature_1), delta)
    # y = np.arange(min(feature_2), max(feature_2), delta)

    # make a numpy arange from the minimum feature until the maximum features
    # delta is the size of spacing between samples
    delta = 0.1
    x = np.arange(-2.0, 4.0, delta)
    y = np.arange(-3.0, 4.0, delta)

    # make a 2-D grind
    x, y = np.meshgrid(x, y)

    # assign bivariate Gaussian distribution for equal shape X, Y.
    z1 = mlab.bivariate_normal(x, y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0)
    z2 = mlab.bivariate_normal(x, y, sigmax=1.5, sigmay=0.5, mux=1, muy=1)
    z = 10.0 * (z2 - z1)

    # create plot
    contour_plot = plt.contour(x, y, z)
    # assign labels and title
    plt.clabel(contour_plot, inline=1, fontsize=10)
    plt.title('Feature 1 against feature 2')
    plt.savefig("./graphs/figures/f1_2.png")


def initialization_based(input_array):
    """ Return the data as one hot encoded data

    input_array: numpy array, labels of the input data
    """

    # search for the unique labels in the array
    oh_array = np.unique(input_array, return_inverse=True)[1]
    # set the predicted class on 1, and all the other classes on 0
    out = np.zeros((oh_array.shape[0], oh_array.max() + 1), dtype=int)
    out[np.arange(out.shape[0]), oh_array.ravel()] = 1
    return out


def data_parser(data):
    """ Return the labels and the features as two separated numpy arrays

    data: closed .txt file, contains 100 samples with two features and 1 label
    """

    with open(data, 'r') as inp:

        # take every sample
        # the last line in the text file is empty, so reading until -1
        samples = inp.read().split('\n')[:-1]

        vec = []
        labels = []
        for sample in samples:
            # file is tab delimited
            split_samples = sample.split('\t')
            # last column contains the label
            labels.append(int(split_samples[-1]))

            features = []
            for feature in split_samples[:-1]:
                features.append(float(feature))
            vec.append(features)

        # make the features and labels as a numpy array
        vec = np.array(vec)
        labels = np.array(labels)
        return vec, labels


if __name__ == "__main__":
    data = "../../Thesis/data/twoclass.txt"
    vec, labels = data_parser(data)
    labels = initialization_based(labels)
    neural_network(vec, labels)
    plot(vec)
