import numpy as np
import h5py


def h_sigmoid(Z):
    """
    Implementation of the Sigmoid function.
    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    Z_value -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    Z_value = Z

    assert(A.shape == Z.shape)

    return A, Z_value


def h_relu(Z):
    """
    Implementation of the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    Z_value -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)
    Z_value = Z

    assert(A.shape == Z.shape)

    return A, Z_value


def h_reluGradient(dA, Z_value):
    """
    Implementation of the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z_value -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = Z_value
    dZ = np.array(dA, copy=True)

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (
        dZ.shape == Z.shape), f"shape of dA and Z doesn't match \nShape of dA : {dA.shape},\nShape of Z : {Z.shape}"

    return dZ


def h_sigmoidGradient(dA, Z_value):
    """
    Implementation of the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z_value -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = Z_value

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (
        dZ.shape == Z.shape), f"shape of dA and Z doesn't match \nShape of dA : {dA.shape},\nShape of Z : {Z.shape}"

    return dZ


def load_data(path):
    """
    path : path to the dataset.
    """
    train_dataset = h5py.File(str(path) + r'\datasets\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][
        :])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][
        :])  # your train set labels

    test_dataset = h5py.File(str(path) + r'\datasets\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][
        :])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][
                               :])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
