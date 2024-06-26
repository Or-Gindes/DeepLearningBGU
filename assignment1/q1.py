"""
DeepLearning Assignment1 - Implementation of a simple neural network “from scratch”
Authors: Or Gindes & Roei Zaady
"""
import numpy as np
from typing import Dict, Tuple, List

np.random.seed(2)
EPSILON = 1e-6


# 1.a
def initialize_parameters(layer_dims: List) -> Dict[str, List[np.ndarray]]:
    """
    :param layer_dims:  an array of the dimensions of each layer in the network
                        (layer 0 is the size of the flattened input, layer L is the output softmax)
    :return: a dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL)
    """
    params = {
        f"layer_{i}": [
            np.random.randn(layer_dims[i + 1], layer_dims[i]) / 10,  # Weights
            np.zeros((layer_dims[i + 1], 1))  # Biases
        ] for i in range(len(layer_dims) - 1)
    }
    return params


# 1.b
def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) \
        -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
     linear part of a layer's forward propagation
    :param A: The activations of the previous layer
    :param W: The weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    :param b: The bias vector of the current layer (of shape [size of current layer, 1])
    :return: Z – linear component of the activation function (i.e., the value before applying the non-linear function)
             linear_cache – a tuple containing A, W, b (stored for making the backpropagation easier to compute)
    """
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)
    return Z, linear_cache


# 1.c
def softmax(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param Z:   The linear component of the activation function
    :return:    A – the softmax activations of the layer
                activation_cache – returns Z, which will be useful for the backpropagation
    """
    A = np.exp(Z) / sum(np.exp(Z))
    return A, Z


# 1.d
def relu(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param Z:   The linear component of the activation function
    :return:    A – the relu activations of the layer
                activation_cache – returns Z, which will be useful for the backpropagation
    """
    A = np.maximum(0, Z)
    return A, Z


# 1.e
def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str) -> Tuple:
    """
    forward propagation for the LINEAR->ACTIVATION layer
    :param A_prev: The activations of the previous layer
    :param W: The weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    :param b: The bias vector of the current layer (of shape [size of current layer, 1])
    :param activation: The activation function to be used (a string, either “softmax” or “relu”)
    :return: A – the activations of the current layer
             cache – a joint dictionary containing both linear_cache and activation_cache
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "softmax":
        A, activation_cache = softmax(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    else:
        raise ValueError(f"activation {activation} isn't implemented")
    cache = [linear_cache, activation_cache]
    return A, cache


# 1.f
def L_model_forward(X: np.ndarray, parameters: Dict, use_batchnorm: bool) -> Tuple[np.ndarray, List]:
    """
    forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation
    :param X: the data, numpy array of shape (input size, number of examples)
    :param parameters: the initialized W and b parameters of each layer
    :param use_batchnorm: a boolean flag used to determine whether to apply batchnorm after the activation
    :return: AL – the last post-activation value
             caches – a list of all the cache objects generated by the linear_forward function
    """
    A_prev = X
    caches = []
    for n_layer, layer_params in enumerate(parameters.values()):
        W, b = layer_params
        activation = "softmax" if n_layer == len(parameters) - 1 else "relu"
        A, cache = linear_activation_forward(A_prev=A_prev, W=W, b=b, activation=activation)
        caches.append(cache)
        if use_batchnorm and activation == "relu":
            A = apply_batchnorm(A)
        A_prev = A

    AL = A
    return AL, caches


# 1.g
def compute_cost(AL: np.ndarray, Y: np.ndarray, parameters: Dict, l2_regularization: bool = False) -> float:
    """
    Compute cost function categorical cross-entropy loss
    :param AL: probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
    :param Y: the labels vector (i.e. the ground truth)
    :param parameters: the W and b parameters of each layer
    :param l2_regularization: boolean - indicates if l2 regularization should be used
    :return: cost – the cross-entropy cost
    """
    n_examples = Y.shape[1]
    cost = -1 / n_examples * sum(np.sum(np.multiply(Y, np.log(AL + EPSILON)), axis=0))

    # l2 Norm
    if l2_regularization:
        l2_cost = sum([np.sum(np.square(layer[0])) for layer in parameters.values()])
        l2_cost = (EPSILON / (2 * n_examples)) * l2_cost
        cost += l2_cost
    return cost


# 1.h
def apply_batchnorm(A: np.ndarray) -> np.ndarray:
    """
    Performs batchnorm on the received activation values of a given layer
    :param A: The activation values of a given layer
    :return: NA - the normalized activation values, based on the formula learned in class

    """
    NA = (A - np.mean(A)) / np.sqrt(np.var(A) + EPSILON)
    return NA
