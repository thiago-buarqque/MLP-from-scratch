from typing import Union

import numpy as np

from MultilayerPerceptron import ActivationFunctions


class Layer:
    def __init__(self,
                 neurons: int,
                 func: str = 'sigmoid',
                 initial_weights: Union[dict[float], None] = None,
                 input_dim=None,
                 next_layer=None
                 ):
        self.input_dim = input_dim
        self.output_dim = neurons

        self.forward_pass_input = []
        self.layer_output = []
        self.deltas = []

        if input_dim is not None:
            self.set_input_dim(input_dim)
            self.set_input_dim(input_dim)

        self.weights = []
        if initial_weights is not None:
            self.weights = [initial_weights]

        self.activation_function = ActivationFunctions.sigmoid
        self.activation_derivative = ActivationFunctions.sigmoid_derivative

        self.next_layer = next_layer
        self.layer_name = ''

        if func == 'sigmoid':
            self.activation_function = ActivationFunctions.sigmoid
        elif func == 'relu':
            self.activation_function = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative
        else:
            raise ValueError("Invalid activation function.")

    def set_input_dim(self, input_dim):
        if len(self.deltas) == 0:
            self.deltas.append(list(np.zeros((self.output_dim, input_dim))))

        if len(self.weights) == 0:
            self.weights = [np.random.uniform(-2, 2, (self.output_dim, input_dim))]
            self.generate_biases()

    def generate_biases(self):
        for i in range(self.output_dim):
            self.weights.append(np.random.uniform(-1, 1))
            self.deltas.append(0)

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def feed_layer(self, input_data):
        if len(input_data) != self.input_dim:
            raise TypeError("Input data does not have the same dimension as layer input.")

        self.forward_pass_input = input_data

        weights = self.weights[0]
        biases = self.weights[1:len(self.weights)]

        next_layer_input = np.zeros(self.output_dim)
        # Calculating weights multiplication
        for i, in_data in enumerate(input_data):
            for j in range(self.output_dim):
                next_layer_input[j] += in_data * weights[j][i]

        # Adding biases
        for j in range(len(next_layer_input)):
            next_layer_input[j] += biases[j]

        for i in range(len(next_layer_input)):
            next_layer_input[i] = self.activation_function(next_layer_input[i])

        self.layer_output = next_layer_input

        return next_layer_input

    def get_weights(self):
        # return np.array(self.weights[0]).ravel()
        return self.weights[0]

    def get_weights_and_biases(self):
        return self.weights[0], self.weights[1:len(self.weights)]

    def __str__(self):
        return f"Layer #{self.layer_name} ({self.input_dim}, {self.output_dim})"
