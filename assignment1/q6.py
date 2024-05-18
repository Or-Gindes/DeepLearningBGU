"""
DeepLearning Assignment1 - Implementation of a simple neural network “from scratch”
Authors: Or Gindes & Roei Zaady
"""

import os
import timeit
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from q4 import get_mnist
from q3 import L_layer_model, Predict
import matplotlib.pyplot as plt
import pandas as pd

os.environ["KERAS_BACKEND"] = "torch"

np.random.seed(0)
def main():
    x_train, x_test, y_train, y_test = get_mnist()
    input_dim = x_train.shape[0]

    # Q4.b Batch Normalization = True
    layer_dims = [input_dim, 20, 7, 5, 10]  # 4 layers (aside from the input layer), with the following sizes: 20,7,5,10

    start_time_train = timeit.default_timer()
    parameters, costs, accuracy_histories = L_layer_model(
                                                            X=x_train,
                                                            Y=y_train,
                                                            layer_dims=layer_dims,
                                                            learning_rate=0.009,  # Use a learning rate of 0.009
                                                            num_iterations=100000,
                                                            batch_size=32,
                                                            use_batchnorm=False,
                                                            l2_regularization=True
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

    print('Test duration: ', str(round(elapsed_time, 3)), '\nTotal duration: ', str(round(end_time - start_time_train, 3)))

    n_layers = len(layer_dims) - 1
    fig, axes = plt.subplots(1, n_layers, figsize=(20, 5))
    fig.suptitle('Distribution of Weight Size by Layer - Without L2 Regularization')

    weights_mse = []
    for i in range(n_layers):
        layer = f'layer_{i}'
        weights = parameters[layer][0].flatten()
        weights_mse.append(np.mean(np.square(weights)))
        ax = axes[i]  # plot weight by layer
        ax.hist(weights, bins=30, edgecolor='black')
        ax.set_title(f'Layer {i + 1} Weights')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    print(pd.DataFrame({'layer': np.arange(n_layers) + 1, 'mse': weights_mse}))


if __name__ == "__main__":
   main()