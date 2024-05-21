"""
DeepLearning Assignment1 - Implementation of a simple neural network “from scratch”
Authors: Or Gindes & Roei Zaady
"""
from typing import List, Dict
import numpy as np
from sklearn.model_selection import train_test_split
from q1 import initialize_parameters, L_model_forward, compute_cost
from q2 import L_model_backward, update_parameters

np.random.seed(2)
EPSILON = 1e-6


# 3.a
def L_layer_model(
        X: np.ndarray,
        Y: np.array,
        layer_dims: List,
        learning_rate: float,
        num_iterations: int,
        batch_size: int = 128,
        use_batchnorm: bool = False,
        l2_regularization: bool = False
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
    :param use_batchnorm: boolean - indicates if batch normalization should be used
    :param l2_regularization: boolean - indicates if l2 regularization should be used
    :return: parameters – the parameters learnt by the system during the training
                (the same parameters that were updated in the update_parameters function).
    :return: costs – the values of the cost function (calculated by the compute_cost function).
                One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values).
    :return accuracy_histories - Lists of train and validation accuracy saved after each 100 training iterations
    """
    parameters = initialize_parameters(layer_dims)
    costs = []
    accuracy_histories = {"train": [], "validation": []}

    # Get validation set that is 20% of the training set - the split is stratified by the labels
    X_train, X_val, Y_train, Y_val = train_test_split(
        X.T, Y.T, test_size=0.2, random_state=42, stratify=Y.T
    )
    X_train, X_val, Y_train, Y_val = X_train.T, X_val.T, Y_train.T, Y_val.T

    # Training loop
    num_examples = X_train.shape[1]
    i = 0  # iteration counter
    patience, patience_counter = 20, 0
    epoch_counter = 1
    best_validation_accuracy = 0
    while i < num_iterations:
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            X_batch, Y_batch = X_train[:, batch_start: batch_end], Y_train[:, batch_start: batch_end]

            AL, caches = L_model_forward(X_batch, parameters, use_batchnorm=use_batchnorm)
            grads = L_model_backward(AL, Y_batch, caches, l2_regularization)
            parameters = update_parameters(parameters, grads, learning_rate)

            if i % 100 == 0:
                cost = compute_cost(AL, Y_batch, parameters, l2_regularization)
                costs.append(cost)
                train_accuracy = Predict(X_train, Y_train, parameters, use_batchnorm)
                val_accuracy = Predict(X_val, Y_val, parameters, use_batchnorm)

                if val_accuracy > best_validation_accuracy:
                    best_validation_accuracy = val_accuracy
                    patience_counter = 0

                accuracy_histories["train"].append(train_accuracy)
                accuracy_histories["validation"].append(val_accuracy)

                print(
                    f"Epoch_{epoch_counter}/Iteration_{i}: training_cost = {round(cost, 3)}"
                    f" | training_accuracy = {round(train_accuracy * 100, 3)}%"
                    f" | validation_accuracy = {round(val_accuracy * 100, 3)}%"
                    f" | best validation_accuracy = {round(best_validation_accuracy * 100, 3)}%"
                )

                # Check EarlyStopping - if best validation accuracy >> current val accuracy then score isn't improving
                if i > 0 and best_validation_accuracy - val_accuracy > EPSILON and patience_counter > patience:
                    print("Early stopping reached!")
                    return parameters, costs, accuracy_histories
                else:
                    patience_counter += 1

                if i == num_iterations:
                    return parameters, costs, accuracy_histories

            i += 1

        # The inner loop represents a full epoch and when it is complete the epoch counter is increased
        epoch_counter += 1

    return parameters, costs, accuracy_histories


# 3.b
def Predict(X: np.ndarray, Y: np.array, parameters: Dict, use_batchnorm: bool) -> float:
    """
    The function receives input data and the true labels and calculates the accuracy of the trained neural network on
    the data.
    :param X: The input data, a numpy array of shape (height*width, number_of_examples)
    :param Y: The “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: A python dictionary containing the DNN architecture’s parameters
    :param use_batchnorm: boolean - indicates if batch normalization should be used
    :return: accuracy – the accuracy measure of the neural net on the provided data
        (i.e. the percentage of the samples for which the correct label receives the highest confidence score).
    """
    num_examples = X.shape[1]
    AL, _ = L_model_forward(X, parameters, use_batchnorm=use_batchnorm)
    labels, y_pred = np.argmax(Y, axis=0), np.argmax(AL, axis=0)
    accuracy = np.sum(y_pred == labels) / num_examples
    return accuracy
