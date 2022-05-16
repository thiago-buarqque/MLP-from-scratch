import numpy as np

from MultilayerPerceptron import ActivationFunctions


class Layer:
    def __init__(self, input_dim, neurons, func='sigmoid', next_layer=None, initial_weights=None, initial_biases=None):
        self.input_dim = input_dim
        self.output_dim = neurons

        self.forward_pass_input = []
        self.layer_output = []
        self.deltas = []
        self.deltas.append(list(np.zeros((neurons, input_dim))))

        if initial_weights is None:
            self.weights = [np.random.uniform(-2, 2, (neurons, input_dim))]
        else:
            self.weights = [initial_weights]

        # Adding biases
        for i in range(neurons):
            if initial_biases is None:
                self.weights.append(np.random.uniform(-1, 1))
            else:
                self.weights.append(initial_biases[i])
            self.deltas.append(0)

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

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def feed_layer(self, input_data):
        if len(input_data) != self.input_dim:
            raise TypeError("Input data does not have the same dimension as layer input.")

        self.forward_pass_input = input_data

        weights = self.weights[0]
        biases = self.weights[1:len(self.weights)]

        # print(f'Input data: {input_data}')
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
