"""
CS 228: Probabilistic Graphical Models
Winter 2020
Programming Assignment 1: Bayesian Networks

Author: Aditya Grover
Updated by Jiaming Song

The following setup is recommended:
- python >= 3 (for pickle)
- scipy >= 1.4 (for logsumexp)
although code has been provided to handle import errors for earlier versions.
"""
import sys
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

# debug
import time


def plot_histogram(data, title='histogram', xlabel='value', ylabel='frequency', savefile='hist'):
    '''
    Plots a histogram.
    '''

    plt.figure()
    plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savefile, bbox_inches='tight')
    plt.show()
    plt.close()

    return


def get_p_z1(z1_val):
    '''
    Helper. Computes the prior probability for variable z1 to take value z1_val.
    P(Z1=z1_val)
    '''

    return bayes_net['prior_z1'][z1_val]


def get_p_z2(z2_val):
    '''
    Helper. Computes the prior probability for variable z2 to take value z2_val.
    P(Z2=z2_val)
    '''

    return bayes_net['prior_z2'][z2_val]


def get_pixels_sampled_from_p_x_joint_z1_z2():
    '''
    Called by q4()

    This function should sample from the joint probability distribution specified by the model,
    and return the sampled values of all the pixel variables (x).
    Note that this function should return the sampled values of ONLY the pixel variables (x),
    discarding the z part.
    TODO: replace pass with your code to implement the above function
    '''
    # sample from joint distribution, taking advantage of the bayesian network
    # sample from Z_1
    z_1_sample = np.random.choice(
        disc_z1, 1, p=[bayes_net['prior_z1'][disc_z1[i]] for i in range(len(disc_z1))])
    # sample from Z_2
    z_2_sample = np.random.choice(
        disc_z2, 1, p=[bayes_net['prior_z2'][disc_z2[i]] for i in range(len(disc_z2))])

    '''
    inspect data

    print("sample of z_1: ", z_1_sample[0], "p(z_1)", get_p_z1(z_1_sample[0]))
    print("sample of z_2: ", z_2_sample[0], "p(z_2)", get_p_z1(z_2_sample[0]))
    print("p(X_1 = 0 | Z1  , Z2 ): ",
          bayes_net['cond_likelihood'][(z_1_sample[0], z_2_sample[0])][0][0])
    print("max: ", np.max(
        bayes_net['cond_likelihood'][(z_1_sample[0], z_2_sample[0])][0]))
    '''

    # sample from X_1, ..., X_784 given sampled Z_1 and Z_2
    X_sample = []
    for i in range(784):
        probability_Xi_0 = bayes_net['cond_likelihood'][(
            z_1_sample[0], z_2_sample[0])][0][i]
        X_i_sample = np.random.choice(
            [0, 1], 1, p=[1 - probability_Xi_0, probability_Xi_0])
        X_sample.append(X_i_sample)

    return np.array(X_sample)


def get_p_x_cond_z1_z2(z1_val, z2_val):
    '''
    Called by q5()

    Computes the conditional probability of the entire vector x,
    given that z1 assumes value z1_val and z2 assumes value z2_val
    TODO: replace pass with your code to implement the above function
    '''
    expected_X = []
    for i in range(784):
        # expectation for the binary distribution is p
        expected_X.append(
            bayes_net['cond_likelihood'][(z1_val, z2_val)][0][i])
    return np.array(expected_X)


def get_conditional_expectation(data):
    '''
    Called by q7().

    This function should return two values which correspond to the conditional expectation of z_1 and z_2 when x is observed to be data.

    TODO: replace pass with your code to implement the above function
    '''
    mean_z1 = []
    mean_z2 = []
    # calculate conditional probability p( (Z1, Z2) | X )
    # apply bayes rule, -> p(Z1, Z2), X) / p(X)
    for i in range(len(data)):

        log_p_z1_z2_x = []
        z1_list = []
        z2_list = []

        for z1 in disc_z1:
            for z2 in disc_z2:

                z1_list.append(z1)
                z2_list.append(z2)

                log_p_z1_z2_x.append(np.log(get_p_z1(z1)) +
                                     np.log(get_p_z2(z2)) +
                                     np.dot(np.log(bayes_net['cond_likelihood'][(z1, z2)][0]), data[i]) +
                                     np.dot(np.log(np.subtract(1, bayes_net['cond_likelihood'][(z1, z2)][0])), np.subtract(1, data[i])))

        log_p_X = logsumexp(log_p_z1_z2_x)
        p_cond_z1_z2 = np.exp(log_p_z1_z2_x) / np.exp(log_p_X)
        # compute conditional expectation of Z1 and Z2
        mean_z1.append(np.dot(z1_list, p_cond_z1_z2))
        mean_z2.append(np.dot(z2_list, p_cond_z1_z2))

    return mean_z1, mean_z2


def q4():
    '''
    Plots the pixel variables sampled from the joint distribution as 28 x 28 images.
    TODO: no need to modify the code here, but implement get_pixels_sampled_from_p_x_joint_z1_z2()
    '''

    plt.figure()
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(get_pixels_sampled_from_p_x_joint_z1_z2().reshape(
            28, 28), cmap='gray')
        plt.title('Sample: ' + str(i+1))
    plt.tight_layout()
    plt.savefig('a4', bbox_inches='tight')
    plt.show()
    plt.close()

    return


def q5():
    '''
    Plots the expected images for each latent configuration on a 2D grid.
    TODO: no need to modify the code here, but implement get_p_x_cond_z1_z2()
    '''

    canvas = np.empty((28*len(disc_z2), 28*len(disc_z1)))
    for i, z1_val in enumerate(disc_z1):
        for j, z2_val in enumerate(disc_z2):
            canvas[(len(disc_z2)-j-1)*28:(len(disc_z2)-j)*28, i*28:(i+1)
                   * 28] = get_p_x_cond_z1_z2(z1_val, z2_val).reshape(28, 28)

    plt.figure()
    plt.imshow(canvas, cmap='gray')
    plt.xlabel('Z_1')
    plt.ylabel('Z_2')
    plt.tight_layout()
    plt.savefig('a5', bbox_inches='tight')
    plt.show()
    plt.close()

    return


# helper function for inspecting data
def draw_plot(data):
    plt.figure()
    plt.imshow(data.reshape(28, 28), cmap='gray')
    plt.show()
    plt.close()


def q6():
    '''
    The provided code loads the data and plots the histogram.
    '''
    mat = loadmat('q6.mat')
    val_data = mat['val_x']
    test_data = mat['test_x']

    '''
    TODO: populate two lists, `real_marginal_log_likelihood` and `corrupt_marginal_log_likelihood` which contains the log-likelihood of the "real" and "corrupted" samples in the test data.

    You might want to use the logsumexp function from scipy (imported from above).
    '''
    # google doc: https://docs.google.com/document/d/1UHAPn1i4W5nhUMXIy8wQbtO8YhnTKggbEw8nBXwQKxQ/edit

    # inspect data
    # print("val_data size: ", len(val_data))
    # draw_plot(val_data[0])
    # print("test_data size: ", len(test_data))
    # draw_plot(test_data[0])

    # compute mean and std deviation of marginal log-likelihood on validation set
    val_data_size = len(val_data)
    val_log_p_X = np.empty(val_data_size)
    for i in range(val_data_size):
        p_z1_z2_x = []
        for z1 in disc_z1:
            for z2 in disc_z2:
                p_z1_z2_x.append(np.log(get_p_z1(z1)) +
                                 np.log(get_p_z2(z2)) +
                                 np.dot(np.log(bayes_net['cond_likelihood'][(z1, z2)][0]), val_data[i]) +
                                 np.dot(np.log(np.subtract(1, bayes_net['cond_likelihood'][(z1, z2)][0])), np.subtract(1, val_data[i])))
                # print("z1: ", np.log(get_p_z1(z1)))
                # print("z2: ", np.log(get_p_z2(z2)))
                # time.sleep(.5)
                # print("log(p(x | z1, z2)): ", np.log(
                #     sum(bayes_net['cond_likelihood'][(z1, z2)][0])))
                # time.sleep(.5)
                # print("to append: ", np.log(get_p_z1(z1)) + np.log(get_p_z2(z2)) +
                #       sum(np.log(bayes_net['cond_likelihood'][(z1, z2)][0])))
                # time.sleep(.5)
                # print("p_z1_z2_x: ", p_z1_z2_x)

        val_log_p_X[i] = logsumexp(p_z1_z2_x)
        # print("p_z1_z2_x: ", p_z1_z2_x)
        # print("i: ", i, "val_log_p_X:", val_log_p_X[i])

    mean_val_log_p_X = np.mean(val_log_p_X)
    std_val_log_p_X = np.std(val_log_p_X)

    # debug: (intermediate result)
    # print("mean: ", mean_val_log_p_X)
    # print("std_dev: ", std_val_log_p_X)
    # mean_val_log_p_X = -138.6747686882813
    # std_val_log_p_X = 45.4517877152083

    # classify real or corrupt on test set
    test_data_size = len(test_data)
    real_marginal_log_likelihood = []
    corrupt_marginal_log_likelihood = []
    for i in range(test_data_size):
        p_z1_z2_x = []
        for z1 in disc_z1:
            for z2 in disc_z2:
                p_z1_z2_x.append(np.log(get_p_z1(z1)) +
                                 np.log(get_p_z2(z2)) +
                                 np.dot(np.log(bayes_net['cond_likelihood'][(z1, z2)][0]), test_data[i]) +
                                 np.dot(np.log(np.subtract(1, bayes_net['cond_likelihood'][(z1, z2)][0])), np.subtract(1, test_data[i])))
        test_log_p_X = logsumexp(p_z1_z2_x)

        if(abs(test_log_p_X - mean_val_log_p_X) <= 3 * std_val_log_p_X):
            real_marginal_log_likelihood.append(test_log_p_X)
        else:
            corrupt_marginal_log_likelihood.append(test_log_p_X)

    # print("# real: ", len(real_marginal_log_likelihood))

    plot_histogram(real_marginal_log_likelihood, title='Histogram of marginal log-likelihood for real data',
                   xlabel='marginal log-likelihood', savefile='a6_hist_real')

    plot_histogram(corrupt_marginal_log_likelihood, title='Histogram of marginal log-likelihood for corrupted data',
                   xlabel='marginal log-likelihood', savefile='a6_hist_corrupt')

    return


def q7():
    '''
    Loads the data and plots a color coded clustering of the conditional expectations.

    TODO: no need to modify code here, but implement the `get_conditional_expectation` function according to the problem statement.
    '''

    mat = loadmat('q7.mat')
    data = mat['x']
    labels = mat['y']
    labels = np.reshape(labels, [-1])

    mean_z1, mean_z2 = get_conditional_expectation(data)

    plt.figure()
    plt.scatter(mean_z1, mean_z2, c=labels)
    plt.xlabel('Z_1')
    plt.ylabel('Z_2')
    plt.colorbar()
    plt.grid()
    plt.savefig('a7', bbox_inches='tight')
    plt.show()
    plt.close()

    return


def load_model(model_file):
    '''
    Loads a default Bayesian network with latent variables (in this case, a variational autoencoder)
    '''

    with open('trained_mnist_model', 'rb') as infile:
        if sys.version_info.major > 2:
            # add try-catch to resolve error in loading model
            # as suggested in https://piazza.com/class/k4bfe1meair4sg?cid=24
            try:
                cpts = pkl.load(infile, encoding='latin1')
            except ImportError:
                data = open('trained_mnist_model').read().replace('\r\n', '\n')
                cpts = pkl.loads(data.encode('latin1'), encoding='latin1')
        else:
            cpts = pkl.load(infile)

    model = {}
    model['prior_z1'] = cpts[0]
    model['prior_z2'] = cpts[1]
    model['cond_likelihood'] = cpts[2]

    return model


def main():
    global disc_z1, disc_z2
    n_disc_z = 25
    disc_z1 = np.linspace(-3, 3, n_disc_z)
    disc_z2 = np.linspace(-3, 3, n_disc_z)

    global bayes_net
    bayes_net = load_model('trained_mnist_model')

    '''
    TODO: Using the above Bayesian Network model, complete the following parts.
    '''
    # Your code should save a figure named a4
    q4()

    # Your code should save a figure named a5
    q5()

    # Your code should save two figures starting with a6
    q6()

    # Your code should save a figure named a7
    q7()

    return


if __name__ == '__main__':
    main()
