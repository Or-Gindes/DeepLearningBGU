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
        use_batchnorm=True  # Activate the batchnorm option at this point
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
              use_batchnorm=True))
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time_test

    print('Test duration: ', str(round(elapsed_time, 3)), '\nTotal duration: ',
          str(round(end_time - start_time_train, 3)))


if __name__ == "__main__":
    main()
