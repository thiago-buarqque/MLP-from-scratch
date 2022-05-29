import math

import numpy as np
from sklearn.metrics import log_loss


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def param_moving_avg(last_step_size, delta):
    return (0.99 * last_step_size) + ((1 - 0.99) * (delta ** 2))


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
            
            self.layers[i].layer_name = f'{i}'
        self.layers[-1].layer_name = f'{len(self.layers) - 1}'

    def get_layers(self):
        return self.layers

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
                    # The neuron output relative error
                    next_layer_relative_error = 0

                    next_layer_weights = next_layer.get_weights()
                    next_layer_deltas = next_layer.deltas[0]

                    for l in range(len(next_layer_weights)):
                        next_layer_relative_error += next_layer_deltas[l][j] * next_layer_weights[l][j]

                    for k, neuron_weights in enumerate(neuron):
                        # How much the output of h_i change with respect the neuron input
                        neuron_input_delta = current_layer.activation_derivative(current_layer_output[j])

                        # Calcualting weight delta

                        # The "How much the total neuron input changes with respect to w_i" value
                        # is calculated when updating the parameter. This is just the output
                        # from previous layer related to w_i
                        current_layer.deltas[0][j][k] = neuron_input_delta * next_layer_relative_error

                    # Calcualting bias delta
                    # current_layer.deltas[j + 1] = current_layer.activation_derivative(current_layer_output[i])
                    current_layer.deltas[j + 1] = current_layer.deltas[0][j][0]
            else:
                current_layer_output = current_layer.layer_output
                for j, neuron in enumerate(current_layer_weights):
                    for k, neuron_weights in enumerate(neuron):
                        # How much the error change with respect to the output
                        output_delta = current_layer_output[j] - expected_output[j]

                        # How much the output of o_i change with respect the neuron input
                        neuron_input_delta = current_layer.activation_derivative(current_layer_output[j])

                        # Calculating weight delta

                        # The "How much the total neuron input changes with respect to w_i" value
                        # is calculated when updating the parameter. This is just the output
                        # from previous layer related to w_i
                        current_layer.deltas[0][j][k] = output_delta * neuron_input_delta

                    # Calculating bias delta
                    # current_layer.deltas[j + 1] = current_layer.activation_derivative(current_layer_output[j])
                    current_layer.deltas[j + 1] = current_layer.deltas[0][j][0]

    def update_params(self):
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            forward_pass_input = self.layers[i].forward_pass_input

            weights, biases = layer.get_weights_and_biases()

            # Updating weights
            for j, neuron in enumerate(weights):
                for k in range(len(neuron)):
                    param_last_step_size = layer.moving_avg[0][j][k]

                    # forward_pass_input[k] is "How much the total neuron input changes with respect to w_i"
                    param_gradient = layer.deltas[0][j][k] * forward_pass_input[k]

                    new_moving_avg = param_moving_avg(
                        param_last_step_size,
                        param_gradient
                    )

                    layer.moving_avg[0][j][k] = new_moving_avg

                    step_size = self.lr / (1e-8 + math.sqrt(new_moving_avg))

                    neuron[k] = neuron[k] - (step_size * param_gradient)

            # Updating biases
            for j in range(len(biases)):
                param_last_step_size = layer.moving_avg[j + 1]

                param_gradient = layer.deltas[j + 1]

                new_moving_avg = param_moving_avg(
                    param_last_step_size,
                    param_gradient
                )

                layer.moving_avg[j + 1] = new_moving_avg

                step_size = self.lr / (1e-8 + math.sqrt(new_moving_avg))

                biases[j] = biases[j] - (step_size * param_gradient)

    def optimize(self, x, y, epochs):
        if len(x) == 0:
            raise ValueError('No data provided.')
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
                  f' Accuracy: {accuracy_metric(np.array(y).ravel(), np.array(predictions).ravel())}')
                  # f' --- Preds: {np.array(predictions).ravel()}')
