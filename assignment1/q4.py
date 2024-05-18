"""
DeepLearning Assignment1 - Implementation of a simple neural network “from scratch”
Authors: Or Gindes & Roei Zaady
"""
import os
import timeit
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from q3 import L_layer_model, Predict
import matplotlib.pyplot as plt
import pandas as pd

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

    # Q4.b Batch Normalization = False
    layer_dims = [input_dim, 20, 7, 5, 10]  # 4 layers (aside from the input layer), with the following sizes: 20,7,5,10

    start_time_train = timeit.default_timer()
    parameters, costs, accuracy_histories = L_layer_model(
        X=x_train,
        Y=y_train,
        layer_dims=layer_dims,
        learning_rate=0.009,  # Use a learning rate of 0.009
        num_iterations=100000,
        batch_size=32,
        use_batchnorm=False  # Do not activate the batchnorm option at this point
    )
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time_train
    print('Training duration: ', str(round(elapsed_time, 3)))

    start_time_test = timeit.default_timer()
    print('Testing Accuracy: ',
          Predict(
              X=x_test,
              Y=y_test,
              parameters=parameters,
              use_batchnorm=False))
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time_test

    print('Test duration: ', str(round(elapsed_time, 3)), '\nTotal duration: ',
          str(round(end_time - start_time_train, 3)))

    n_layers = len(layer_dims) - 1
    fig, axes = plt.subplots(1, n_layers, figsize=(20, 5))
    fig.suptitle('Distribution of Weight Size by Layer - Without L2 Regularization')

    weights_means = []
    for i in range(n_layers):
        layer = f'layer_{i}'
        weights = parameters[layer][0].flatten()
        weights_means.append(np.mean(np.square(weights)))
        ax = axes[i]  # plot weight by layer
        ax.hist(weights, bins=30, edgecolor='black')
        ax.set_title(f'Layer {i + 1} Weights')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    print(pd.DataFrame({'layer': np.arange(n_layers) + 1, 'mse': weights_means}))


if __name__ == "__main__":
    main()
