"""
DeepLearning Assignment1 - Implementation of a simple neural network “from scratch”
Authors: Or Gindes & XXXX
"""
from typing import List, Dict
import numpy as np
from q1 import initialize_parameters, L_model_forward, compute_cost
from q2 import L_model_backward, update_parameters

np.random.seed(42)


# 3.a
def L_layer_model(
        X: np.ndarray,
        Y: np.array,
        layer_dims: List,
        learning_rate: float,
        num_iterations: int = 3000,
        batch_size: int = 64
):
    """
    Implements a L-layer neural network. All layers but the last should have the ReLU activation function,
    and the final layer will apply the softmax activation function.
    The size of the output layer should be equal to the number of labels in the data.
    the function should use the earlier functions in the following order:
        initialize -> L_model_forward -> compute_cost -> L_model_backward -> update parameters

    :param X: The input data, a numpy array of shape (height*width , number_of_examples)
    :param Y: The “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param layer_dims: A list containing the dimensions of each layer, including the input
    :param learning_rate: the learning rate used to update the parameters (the “alpha”)
    :param num_iterations: The number of training iterations to perform
    :param batch_size: the number of examples in a single training batch
    :return: parameters – the parameters learnt by the system during the training
                (the same parameters that were updated in the update_parameters function).
    :return: costs – the values of the cost function (calculated by the compute_cost function).
                One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values).
    """
    parameters = initialize_parameters(layer_dims)
    costs = []
    num_examples = X.shape[1]

    i = 0   # iteration counter
    while i < num_iterations:
        # epoch loop
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            X_batch, Y_batch = X[:, batch_start: batch_end], Y[:, batch_start: batch_end]

            AL, caches = L_model_forward(X_batch, parameters, use_batchnorm=True)
            cost = compute_cost(AL, Y_batch)
            grads = L_model_backward(AL, Y_batch, caches)
            parameters = update_parameters(parameters, grads, learning_rate)

            if i % 100 == 0:
                costs.append(cost)

            i += 1

    return parameters, costs


# 3.b
def Predict(X: np.ndarray, Y: np.array, parameters: Dict) -> float:
    """
    The function receives input data and the true labels and calculates the accuracy of the trained neural network on
    the data.
    :param X: The input data, a numpy array of shape (height*width, number_of_examples)
    :param Y: The “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: A python dictionary containing the DNN architecture’s parameters
    :return: accuracy – the accuracy measure of the neural net on the provided data
        (i.e. the percentage of the samples for which the correct label receives the highest confidence score).
    """
    num_examples = X.shape[1]
    AL, _ = L_model_forward(X, parameters, use_batchnorm=False)
    labels, y_pred = np.argmax(Y, axis=0), np.argmax(AL, axis=0)
    accuracy = np.sum(y_pred == labels) / num_examples
    return accuracy

