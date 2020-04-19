import numpy as np
import matplotlib.pyplot as plt
import h5py
import deepLnn

class NeuralNets:
    
    def __init__(self,layer_dims):
        """
        The top level class of this framework.
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        """

        self.layer_dims = layer_dims
        self.parameters = {}
        self.y_hat = None
        self.grads = None

    def initializeParameters(self):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        
        self.parameters = deepLnn.initialize_parameters_deep(self.layer_dims)
        return self.parameters
    

    def L_model_forward(self,X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        A = X
        # number of layers in the neural network
        L = len(self.parameters) // 2

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A, cache = deepLnn.linear_activation_forward(
                A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], activation="relu")
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = deepLnn.linear_activation_forward(
            A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], activation="sigmoid")
        caches.append(cache)

        assert(AL.shape == (1, X.shape[1]))

        self.y_hat = AL
        return AL, caches

    def L_model_backward(self,AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

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
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches".
        # Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)
                                                              ] = deepLnn.linear_activation_backward(dAL, current_cache, activation="sigmoid")

        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = deepLnn.linear_activation_backward(
                grads["dA" + str(l + 1)], current_cache, activation="relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        self.grads = grads
        return grads


    def update_parameters(self,learning_rate):
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

    def predict(self,X, y):
        """
        This function is used to predict the results of a  L-layer neural network.

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
        probas, caches = self.L_model_forward(X)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        # print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        print("Accuracy: " + str(np.sum((p == y) / m)))

        return p

    def L_layer_model(self,X, Y, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
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
        
        # Parameters initialization. (≈ 1 line of code)
        ### START CODE HERE ###
        parameters = self.initializeParameters()
        ### END CODE HERE ###
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = self.L_model_forward(X)
            ### END CODE HERE ###
            
            # Compute cost.
            ### START CODE HERE ### (≈ 1 line of code)
            cost = deepLnn.compute_cost(AL, Y)
            ### END CODE HERE ###
        
            # Backward propagation.
            ### START CODE HERE ### (≈ 1 line of code)
            grads = self.L_model_backward(AL, Y, caches)
            ### END CODE HERE ###
     
            # Update parameters.
            ### START CODE HERE ### (≈ 1 line of code)
            parameters = self.update_parameters(learning_rate)
            ### END CODE HERE ###
                    
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        self.parameters = parameters
        return parameters


if __name__ == "__main__":
    train_x_orig, train_y, test_x_orig, test_y, classes = deepLnn.load_data()
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
    layer4Model = NeuralNets(layers_dims)
    layer4Model.L_layer_model(train_x, train_y, num_iterations = 2500, print_cost = True)
    print("Training set Accuracy")
    pred_train = layer4Model.predict(train_x, train_y)
    print("Testing set Accuracy")
    pred_test = layer4Model.predict(test_x, test_y)