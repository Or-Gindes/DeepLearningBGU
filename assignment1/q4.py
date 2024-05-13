"""
DeepLearning Assignment1 - Implementation of a simple neural network “from scratch”
Authors: Or Gindes & XXXX
"""
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from q3 import L_layer_model

os.environ["KERAS_BACKEND"] = "torch"
from keras_core.datasets import mnist

np.random.seed(0)


# 4.a
def get_mnist():
    """
    Get MNIST data and preprocess input
    :return: X,y for train and test sets
    """
    # Load dataset
    (x_train_val, y_train_val), (x_test, y_test) = mnist.load_data()
    # normalize dataset by 255 (max value)
    x_train_val = x_train_val / 255.0
    x_test = x_test / 255.0

    # Flatten the input of train_val and test sets
    img_h, img_w = x_train_val.shape[1], x_train_val.shape[2]
    x_train = x_train_val.reshape((x_train_val.shape[0], img_h * img_w), order='F').T
    x_test = x_test.reshape((x_test.shape[0], img_h * img_w), order='F').T

    # Apply onehot encoding to y
    onehot = OneHotEncoder(sparse_output=False)
    y_train = onehot.fit_transform(y_train_val.reshape(-1, 1)).T
    y_test = onehot.transform(y_test.reshape(-1, 1)).T

    return x_train, x_test, y_train, y_test


def main():
    x_train, x_test, y_train, y_test = get_mnist()
    input_dim = x_train.shape[0]

    # Q4.b
    layer_dims = [input_dim, 20, 7, 5, 10]  # 4 layers (aside from the input layer), with the following sizes: 20,7,5,10
    L_layer_model(
        X=x_train,
        Y=y_train,
        layer_dims=layer_dims,
        learning_rate=0.009,                # Use a learning rate of 0.009
        num_iterations=100000,
        batch_size=32,
        use_batchnorm=False                 # Do not activate the batchnorm option at this point
    )


if __name__ == "__main__":
    main()
