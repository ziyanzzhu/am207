import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def generate_exp1_test_data():

    blob1 = np.random.multivariate_normal([-3, -3], 0.5 * np.eye(2), 75)
    blob2 = np.random.multivariate_normal([3, 3], 0.5 * np.eye(2), 75)
    blob0_train = np.random.multivariate_normal([-1, -1], 0.5 * np.eye(2), 25)
    blob1_train = np.random.multivariate_normal([1, 1], 0.5 * np.eye(2), 25)
    test_points = np.vstack((blob1, blob2, blob0_train, blob1_train))
    test_points_labels = np.random.randint(2,size=200)
    test_points_labels[150:175] = 0
    test_points_labels[175:] = 1
    return test_points, test_points_labels


def generate_exp1_test_data_moons():

    blob1 = np.random.multivariate_normal([0, 1.75], 0.1 * np.eye(2), 75)
    blob2 = np.random.multivariate_normal([1, -1.25], 0.1 * np.eye(2), 75)
    blobs_train, blobs_train_labels = make_moons(n_samples=50, shuffle=False, noise=0.1, random_state=None)
    test_points = np.vstack((blob1, blob2, blobs_train))
    test_points_labels = np.random.randint(2,size=200)
    test_points_labels[150:] = blobs_train_labels
    return test_points, test_points_labels


def plot_training_data(x_train, y_train, samples, test=None):
    
    fig = plt.figure()
    plt.scatter(x_train[:samples,0], x_train[:samples,1], color='blue', alpha=0.5, label='Class 0')
    plt.scatter(x_train[samples:,0], x_train[samples:,1], color='red', alpha=0.5, label='Class 1')
    if test is not None:
        plt.scatter(test[:,0], test[:,1], color='black', alpha=0.5, label='Test Points')
    plt.legend()
    plt.show()
    return None


def generate_nn_list(var_means, var_variance, samples):

    nn_list = []
    for i in range(samples):
        #set random state to make the experiments replicable
        rand_state = 0
        random = np.random.RandomState(rand_state)
        tempNN = Feedforward(architecture, random=random)
        tempNN.weights = np.random.multivariate_normal(var_means, np.diag(var_variance)).reshape((1,var_means.shape[0]))
        nn_list.append(tempNN)

    return nn_list

# Generate a toy dataset for classification with controlled center and spread
def make_blobs(center1, center2, spread1, spread2, samples): 
    ''' input: center and spread of the Gaussian blobs. Samples is the number of class 0 / class 1'''
    class_0 = np.random.multivariate_normal(center1, spread1 * np.eye(2), samples)
    class_1 = np.random.multivariate_normal(center2, spread2 * np.eye(2), samples)
    x_train = np.vstack((class_0, class_1))
    y_train = np.array([0] * samples + [1] * samples)
    return x_train, y_train
