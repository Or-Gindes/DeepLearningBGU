"""
DeepLearning Assignment1 - Implementation of a simple neural network “from scratch”
Authors: Or Gindes & XXXX
"""
import numpy as np
from typing import Dict, Tuple, Literal

np.random.seed(42)


# 1.a
def initialize_parameters(layer_dims: np.array) -> Dict[str, Tuple[np.ndarray, np.array]]:
    """
    :param layer_dims:  an array of the dimensions of each layer in the network
                        (layer 0 is the size of the flattened input, layer L is the output softmax)
    :return: a dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL)
    """
    params = {
        f"layer_{i}": (
            np.random.randn(layer_dims[i], layer_dims[i + 1]),  # Weights
            np.zeros((layer_dims[i + 1], 1))  # Biases
        ) for i in range(len(layer_dims) - 1)
    }
    return params



