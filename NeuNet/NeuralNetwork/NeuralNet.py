import numpy as np
import matplotlib.pyplot as plt
try:
    from HelperFuncs import *
except Exception:
    from NeuNet.NeuralNetwork.HelperFuncs import *


class NeuralNet:

    def __init__(self, layers_dims=None):
        """
        Single Top level class of NeuralNet framwork.
        """
        self.layers_dims = layers_dims
        self.parameters = {}
        self.m = None
        self.y_hat = None
        self.grads = {}
        self.Lambda = 0
        self.hyperInit = 2

    def add_layer(self, n_h):
        """
        Adding a n_h number of nodes layer to the Neural Network.
        Arguments:
        n_h -- number of nodes for the layer.
        """
        self.layers_dims.append(n_h)

    def He_initializeParameters(self):
        """
        Arguments:
        layers_dims -- python array (list) containing the dimensions of each layer in our network
        he_init = False -- if parameters should be He initialized.
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        np.random.seed(1)
        L = len(self.layers_dims)
        for i in range(1, L):
            self.parameters["W" + str(i)] = np.random.randn(
                self.layers_dims[i], self.layers_dims[i - 1]) * np.sqrt(self.hyperInit / self.layers_dims[i - 1])
            self.parameters["b" + str(i)] = np.zeros((self.layers_dims[i], 1))

        return self.parameters

    def random_initializeParameters(self):
        """
        Arguments:
        layers_dims -- python array (list) containing the dimensions of each layer in our network
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        np.random.seed(1)
        L = len(self.layers_dims)
        for i in range(1, L):
            self.parameters["W" + str(i)] = np.random.randn(
                self.layers_dims[i], self.layers_dims[i - 1]) * 0.01
            self.parameters["b" + str(i)] = np.zeros((self.layers_dims[i], 1))

        return self.parameters

    def linear_forward(self, A, W, b):
        """
        Implementation of the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        A_W_b -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        Z = W.dot(A) + b

        assert(Z.shape == (W.shape[0], A.shape[1]))
        A_W_b = (A, W, b)

        return Z, A_W_b

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implementation of the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        A_W_b__Z -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, A_W_b = self.linear_forward(A_prev, W, b)
            A, Z_value = h_sigmoid(Z)

        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, A_W_b = self.linear_forward(A_prev, W, b)
            A, Z_value = h_relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        A_W_b__Z = (A_W_b, Z_value)

        return A, A_W_b__Z

    def forwardProp(self, X):
        """
        Implementation of forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)

        Returns:
        AL -- last post-activation value
        A_W_b__Zs -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        A_W_b__Zs = []
        A = X
        L = len(self.parameters) // 2  # number of layers

        for i in range(1, L):
            A_prev = A

            A, A_W_b__Z = self.linear_activation_forward(A_prev, self.parameters["W" + str(i)],
                                                         self.parameters["b" + str(i)], activation="relu")
            A_W_b__Zs.append(A_W_b__Z)

        AL, A_W_b__Z = self.linear_activation_forward(A, self.parameters["W" + str(L)],
                                                      self.parameters["b" + str(L)], activation="sigmoid")
        A_W_b__Zs.append(A_W_b__Z)

        assert(AL.shape[1] == X.shape[1])

        self.y_hat = AL
        return AL, A_W_b__Zs

    def compute_cost(self, AL, Y):
        """
        Implementation of the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[1]
        self.m = m
        regulCost = 0.0
        if self.Lambda != 0:
            for i in range(1, len(self.layers_dims)):
                regulCost += np.sum(np.square(self.parameters["W" + str(i)]))

        # Compute loss from aL and y.
        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) -
                           np.dot(1 - Y, np.log(1 - AL).T)) + (float(self.Lambda) / (2 * self.m)) * regulCost

        # To make sure your cost's shape is what we expect (e.g. this turns [[17]]
        # into 17).
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        return cost

    def linear_backwardGradient(self, dZ, A_W_b):
        """
        Implementation of the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        A_W_B -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = A_W_b
        m = A_prev.shape[1]

        dW = 1. / m * np.dot(dZ, A_prev.T) + (float(self.Lambda) / self.m) * W
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, A_W_b__Z, activation):
        """
        Implementation of the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        A_W_b__Z -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_W_b, Z = A_W_b__Z

        if activation == "relu":
            dZ = h_reluGradient(dA, Z)
            dA_prev, dW, db = self.linear_backwardGradient(dZ, A_W_b)

        elif activation == "sigmoid":
            dZ = h_sigmoidGradient(dA, Z)
            dA_prev, dW, db = self.linear_backwardGradient(dZ, A_W_b)

        return dA_prev, dW, db

    def backwardProp(self, AL, Y, A_W_b__Zs):
        """
        Implementation of the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                    the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """

        L = len(A_W_b__Zs)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        # initializing the last gradient
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        # sigmoid->linear

        current_A_W_B_Z = A_W_b__Zs[L - 1]
        self.grads["dA" + str(L - 1)], self.grads["dW" + str(L)], self.grads["db" + str(
            L)] = self.linear_activation_backward(dAL, current_A_W_B_Z, activation="sigmoid")

        for i in reversed(range(L - 1)):
            # Relu->Linear gradients
            current_A_W_B_Z = A_W_b__Zs[i]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                self.grads["dA" + str(i + 1)], current_A_W_B_Z, activation="relu")
            self.grads["dA" + str(i)] = dA_prev_temp
            self.grads["dW" + str(i + 1)] = dW_temp
            self.grads["db" + str(i + 1)] = db_temp

        return self.grads

    def update_parameters(self, learning_rate):
        """
        Update parameters using gradient descent

        Returns:
        parameters -- python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...
        """

        L = len(self.parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - \
                learning_rate * self.grads["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - \
                learning_rate * self.grads["db" + str(l + 1)]

        return self.parameters

    def predict(self, X, y, show=False):
        """
        This function is used to predict the results of a L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label.
        y -- true labelled data.

        Returns:
        p -- predictions for the given dataset X.
        """

        m = X.shape[1]
        n = len(self.parameters) // 2  # number of layers in the neural network
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = self.forwardProp(X)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        # print results
        if(show):
            print("predictions: " + str(p))
            print("true labels: " + str(y))
        print("Accuracy: " + str(np.sum((p == y) / m)))

    def fit(self, X, Y, learning_rate=0.0075, num_iterations=3000, Lambda=0, print_cost=False, init="he"):  # lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []                         # keep track of cost
        self.Lambda = Lambda
        if init == "he":
            self.parameters = self.He_initializeParameters()
        else:
            self.parameters = self.random_initializeParameters()

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.forwardProp(X)

            # Compute cost.
            cost = self.compute_cost(AL, Y)

            # Backward propagation.
            self.grads = self.backwardProp(AL, Y, caches)

            # Update parameters.
            self.parameters = self.update_parameters(learning_rate)
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        return self.parameters
