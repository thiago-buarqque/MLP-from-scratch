import math

import numpy as np
from sklearn.metrics import log_loss

from MultilayerPerceptron.Layer import Layer
from MultilayerPerceptron.Neuron import Neuron


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def param_moving_avg(last_step_size, param_gradient):
    return (0.99 * last_step_size) + ((1 - 0.99) * (param_gradient ** 2))


class MLP:
    def __init__(self, lr=0.01, classify_function=None):
        self.input_dim = 0

        self.layers = []
        self.lr = lr

        self.classify_function = classify_function

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
            layer = self.layers[i]
            layer_neurons: [Neuron] = layer.get_neurons()

            # It's a hidden layer
            if i != len(self.layers) - 1:
                next_layer: Layer = self.layers[i + 1]

                layer_output = layer.layer_output
                for j, neuron in enumerate(layer_neurons):
                    # The neuron output relative error
                    next_layer_relative_error = 0

                    next_layer_neurons: [Neuron] = next_layer.get_neurons()

                    for k, _neuron in enumerate(next_layer_neurons):
                        next_layer_relative_error += _neuron.delta * _neuron.weights[j]

                    # How much the output of h_i change with respect the neuron input
                    neuron_input_delta = layer.activation_derivative(layer_output[j])

                    # Calculate weight delta

                    # The "How much the total neuron input changes with respect to w_i" value
                    # is calculated when updating the parameter. This is just the output
                    # from previous layer related to w_i
                    neuron.delta = neuron_input_delta * next_layer_relative_error
            else:
                layer_output = layer.layer_output
                for j, neuron in enumerate(layer_neurons):
                    # How much the error change with respect to the output
                    output_delta = layer_output[j] - expected_output[j]

                    # How much the output of o_i change with respect the neuron input
                    neuron_input_delta = layer.activation_derivative(layer_output[j])

                    # Calculate neuron delta

                    # The "How much the total neuron input changes with respect to w_i" value
                    # is calculated when updating the parameter. This is just the output
                    # from previous layer related to w_i
                    neuron.set_delta(output_delta * neuron_input_delta)

    def update_params(self):
        for i in reversed(range(len(self.layers))):
            layer: Layer = self.layers[i]

            forward_pass_input = layer.get_forward_pass_input()

            neurons: [Neuron] = layer.get_neurons()

            for j, neuron in enumerate(neurons):
                for k in range(len(neuron.weights)):
                    last_moving_avg = neuron.moving_avg[k]

                    if k <= len(forward_pass_input) - 1:
                        # forward_pass_input[j] is "How much the total neuron input changes with respect to w_i"
                        param_gradient = neuron.delta * forward_pass_input[k]
                    else:
                        # It's the neuron bias
                        param_gradient = neuron.delta

                    new_moving_avg = param_moving_avg(
                        last_moving_avg,
                        param_gradient
                    )

                    neuron.moving_avg[k] = new_moving_avg

                    step_size = self.lr / (1e-8 + math.sqrt(new_moving_avg))

                    neuron.weights[k] -= step_size * param_gradient

    def optimize(self, x, y, epochs):
        if len(x) == 0:
            raise ValueError('No data provided.')
        elif len(x[0]) != self.input_dim:
            raise TypeError('Data does not have the same input dimension as the network.')

        for i in range(epochs):
            predictions = []
            raw_predictions = []
            for j, sample in enumerate(x):
                next_layer_input_data = self.evaluate(sample)

                output = self.classify_function(next_layer_input_data)
                predictions.append(output)
                raw_predictions.append(next_layer_input_data)

                self.backward_propagate_error(y[j])
                self.update_params()

            print(f'Epoch={i} Loss: {log_loss(np.array(y).ravel(), np.array(raw_predictions).ravel())}'
                  f' Accuracy: {accuracy_metric(np.array(y).ravel(), np.array(predictions).ravel())}')
                  # f' --- Preds: {np.array(predictions).ravel()}')

    def evaluate(self, x):
        if len(x) != self.input_dim:
            raise TypeError('Data does not have the same input dimension as the network.')

        next_layer_input_data = x
        for k, layer in enumerate(self.layers):
            next_layer_input_data = layer.feed_layer(next_layer_input_data)

        return next_layer_input_data

    def predict(self, x):
        prediction = self.evaluate(x)

        return self.classify_function(prediction)
