from typing import Union

import numpy as np

from MultilayerPerceptron import ActivationFunctions


class Layer:
    def __init__(self,
                 neurons: int,
                 input_dim: int,
                 func: str = 'sigmoid',
                 initial_weights=None
                 ):
        self.input_dim = input_dim
        self.output_dim = neurons

        self.forward_pass_input = None
        self.layer_output = None

        self.weights = None
        self.deltas = [list(np.zeros((self.output_dim, input_dim)))]
        self.moving_avg = [list(np.zeros((self.output_dim, input_dim)))]

        if initial_weights is not None:
            self.weights = [initial_weights]
        else:
            self.initialize_parameters(input_dim)

        self.activation_function = ActivationFunctions.sigmoid
        self.activation_derivative = ActivationFunctions.sigmoid_derivative

        self.next_layer = None
        self.layer_name = ''

        if func == 'relu':
            self.activation_function = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative
        elif func != 'sigmoid':
            raise ValueError("Invalid activation function.")

    def initialize_parameters(self, input_dim):
        self.weights = list([np.random.uniform(-2, 2, (self.output_dim, input_dim))])
        self.generate_biases()

    def generate_biases(self):
        for i in range(self.output_dim):
            # [ ] Change to random initilization. Currently I'm making some tests
            self.weights.append(1)
            self.deltas.append(0)
            self.moving_avg.append(0)

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def feed_layer(self, input_data):
        if len(input_data) != self.input_dim:
            raise TypeError(
                f"Input data does not have the same dimension as layer input. ({len(input_data), self.input_dim})")

        self.forward_pass_input = input_data

        weights = self.weights[0]
        biases = self.weights[1:len(self.weights)]

        layer_output = np.zeros(self.output_dim)

        # Calculating weights multiplication
        for i, in_data in enumerate(input_data):
            for j in range(self.output_dim):
                layer_output[j] += in_data * weights[j][i]

        for j in range(len(layer_output)):
            layer_output[j] = self.activation_function(layer_output[j] + biases[j])

        self.layer_output = layer_output

        return layer_output

    def get_weights(self):
        return self.weights[0]

    def get_weights_and_biases(self):
        return self.weights[0], self.weights[1:len(self.weights)]

    def __str__(self):
        return f"Layer #{self.layer_name} ({self.input_dim}, {self.output_dim})"
