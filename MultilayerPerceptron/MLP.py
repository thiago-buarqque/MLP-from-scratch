import math

import numpy as np
from sklearn.metrics import log_loss


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


class MLP:
    def __init__(self, lr=0.01):
        self.input_dim = 0

        self.layers = []
        self.lr = lr

    def add_layer(self, layer):
        if len(self.layers) == 0:
            self.input_dim = layer.input_dim
        self.layers.append(layer)

        for i in range(len(self.layers) - 1):
            if self.layers[i].next_layer is None:
                self.layers[i].set_next_layer(self.layers[i + 1])
            # self.layers[i + 1].set_input_dim(self.layers[i].output_dim)
            self.layers[i].layer_name = f'{i}'
        self.layers[-1].layer_name = f'{len(self.layers) - 1}'

    def get_layers(self):
        for layer in self.layers:
            print(layer)

    def backward_propagate_error(self, expected_output):
        # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        for i in range(len(self.layers) - 1, -1, -1):
            current_layer = self.layers[i]
            current_layer_weights = current_layer.get_weights()

            # It's a hidden layer
            if i != len(self.layers) - 1:
                next_layer = self.layers[i + 1]

                current_layer_output = current_layer.layer_output
                for j, neuron in enumerate(current_layer_weights):
                    next_layer_relative_error = 0
                    next_layer_weights = next_layer.get_weights()

                    next_layer_deltas = next_layer.deltas[0]
                    for l in range(len(next_layer_weights)):
                        next_layer_relative_error += next_layer_deltas[l][j] * next_layer_weights[l][j]

                    for k, neuron_weights in enumerate(neuron):
                        delta_h_i__net_h_i = current_layer.activation_derivative(current_layer_output[j])

                        # Calcualting weight delta
                        current_layer.deltas[0][j][k] = delta_h_i__net_h_i * next_layer_relative_error
                    # Calcualting bias delta
                    current_layer.deltas[j + 1] -= current_layer.activation_derivative(current_layer_output[i])
            else:
                current_layer_output = current_layer.layer_output
                for j, neuron in enumerate(current_layer_weights):
                    for k, neuron_weights in enumerate(neuron):
                        # EQ1
                        output_delta = -(expected_output[j] - current_layer_output[j])

                        # EQ2
                        neuron_input_delta = current_layer.activation_derivative(current_layer_output[j])

                        # Calculating weight delta
                        # The total net input of o_i change with respect to w_i (EQ3) is used directly in the weight
                        # update so it's easier to calculate the hidden layer delta
                        current_layer.deltas[0][j][k] = output_delta * neuron_input_delta

                    # Calculating bias delta
                    current_layer.deltas[j + 1] -= current_layer.activation_derivative(current_layer_output[j])

    def update_params(self):
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            forward_pass_input = self.layers[i].forward_pass_input

            weights, biases = layer.get_weights_and_biases()

            # Updating weights
            for j, neuron in enumerate(weights):
                for k in range(len(neuron)):
                    param_last_step_size = layer.lrs[0][j][k]

                    # forward_pass_input[k] refer to the EQ3 for both output and hidden layer
                    param_gradient = layer.deltas[0][j][k] * forward_pass_input[k]

                    new_moving_avg = self.calculate_param_step_size(
                        param_last_step_size,
                        param_gradient
                    )

                    layer.lrs[0][j][k] = new_moving_avg

                    step_size = self.lr / (1e-8 + math.sqrt(new_moving_avg))

                    neuron[k] = neuron[k] - (step_size * param_gradient)

            # Updating biases
            for j in range(len(biases)):
                param_last_step_size = layer.lrs[j + 1]

                param_gradient = layer.deltas[j + 1]

                new_moving_avg = self.calculate_param_step_size(
                    param_last_step_size,
                    param_gradient
                )

                layer.lrs[j + 1] = new_moving_avg

                step_size = self.lr / (1e-8 + math.sqrt(new_moving_avg))

                biases[j] = biases[j] - (step_size * param_gradient)

    def optimize(self, x, y, epochs):
        if len(x) == 0:
            raise ValueError('No data.')
        elif len(x[0]) != self.input_dim:
            raise TypeError('Data does not have the same input dimension as the network.')

        for i in range(epochs):
            predictions = []
            for j, sample in enumerate(x):
                next_layer_input_data = sample
                for k, layer in enumerate(self.layers):
                    next_layer_input_data = layer.feed_layer(next_layer_input_data)

                predictions.append(next_layer_input_data)

                self.backward_propagate_error(y[j])
                self.update_params()

            print(f'Epoch={i} Loss: {log_loss(np.array(y).ravel(), np.array(predictions).ravel())}'
                  f' Accuracy: {accuracy_metric(np.array(y).ravel(), np.array(predictions).ravel())}'
                  f' --- Preds: {np.array(predictions).ravel()}')

    def calculate_param_step_size(self, last_step_size, delta):
        return (0.99 * last_step_size) + ((1 - 0.99) * (delta ** 2))
