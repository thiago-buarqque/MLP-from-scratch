import numpy as np

from MultilayerPerceptron import ActivationFunctions


class Layer:
    def __init__(self, input_dim, output_dim, func='sigmoid', next_layer=None):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = [np.random.uniform(-1, 1, (input_dim, output_dim)), np.random.uniform(-1, 1, (output_dim, 1))]
        self.activation_function = ActivationFunctions.sigmoid

        self.next_layer = next_layer

        if func == 'sigmoid':
            self.activation_function = ActivationFunctions.sigmoid
        elif func == 'relu':
            self.activation_function = ActivationFunctions.relu
        else:
            raise ValueError("Invalid activation function.")

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def feed_layer(self, input_data):
        if len(input_data) != self.input_dim:
            raise TypeError("Input data does not have the same dimension as layer input.")

        weights = self.weights[0]
        biases = self.weights[1]

        next_layer_input = np.zeros(self.output_dim)
        for i, in_data in enumerate(input_data):
            for j in range(self.output_dim):
                next_layer_input[j] += in_data * weights[i][j] + biases[j][0]

        for i in range(len(next_layer_input)):
            next_layer_input[i] = self.activation_function(next_layer_input[i])

        return next_layer_input

    def __str__(self):
        return f"Layer ({self.input_dim}, {self.output_dim}) -> {self.next_layer}"
