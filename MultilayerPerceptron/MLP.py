import numpy as np
from sklearn.metrics import log_loss


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


class MLP:
    def __init__(self):
        self.input_dim = 0

        self.layers = []
        self.lr = 0.25

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
        print(f'Printando {len(self.layers)} camadas')
        for layer in self.layers:
            print(layer)

    def backward_propagate_error(self, expected_output):
        # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        for i in range(len(self.layers) - 1, -1, -1):
            curr_layer = self.layers[i]
            # layer_errors = list()
            layer_weights = curr_layer.get_weights()

            # It's a hidden layer
            if i != len(self.layers) - 1:
                next_layer = self.layers[i + 1]

                layer_output = curr_layer.layer_output
                # layer_input = curr_layer.forward_pass_input
                for j, neuron in enumerate(layer_weights):
                    for k, neuron_weights in enumerate(neuron):
                        next_layer_relative_error = 0
                        next_layer_weights = next_layer.get_weights()

                        next_layer_deltas = next_layer.deltas[0]
                        for l in range(len(next_layer_weights)):
                            next_layer_relative_error += next_layer_deltas[l][j] * next_layer_weights[l][j]

                        delta_h_i__net_h_i = curr_layer.activation_derivative(layer_output[j])
                        # delt_net_h_i__delta_w_i = layer_input[j]

                        curr_layer.deltas[0][j][
                            k] = delta_h_i__net_h_i * next_layer_relative_error
                    # Adding bias delta
                    curr_layer.deltas[j + 1] = curr_layer.deltas[i + 1] - curr_layer.activation_derivative(
                        layer_output[i])
            else:
                layer_output = curr_layer.layer_output
                for j, neuron in enumerate(layer_weights):
                    for k, neuron_weights in enumerate(neuron):
                        output_delta = -(expected_output[j] - layer_output[j])
                        neuron_input_delta = curr_layer.activation_derivative(layer_output[j])
                        # Adding weight delta
                        # The total net input of o1 change with respect to w_i is used directly in the weight update
                        # so it's easier to calculate the hidden layer delta
                        curr_layer.deltas[0][j][k] = output_delta * neuron_input_delta
                    # Adding bias delta
                    curr_layer.deltas[j + 1] = curr_layer.deltas[j + 1] - curr_layer.activation_derivative(
                        layer_output[j])

    def update_params(self):
        for i in range(len(self.layers) - 1, -1, -1):
            curr_layer = self.layers[i]
            inputs = self.layers[i].forward_pass_input

            weights, biases = curr_layer.get_weights_and_biases()

            # Updating weights
            for j, neuron in enumerate(weights):
                for k in range(len(neuron)):
                    neuron[k] = neuron[k] - (self.lr * curr_layer.deltas[0][j][k] * inputs[k])

            # Updating biases
            for j in range(len(biases)):
                biases[j] = biases[j] - (self.lr * curr_layer.deltas[j + 1])

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
                # net_pred = 1 if next_layer_input_data > 0.5 else 0

                predictions.append(next_layer_input_data)
                # print(f'Pred for sample {j+1}: {next_layer_input_data}')

                self.backward_propagate_error(y[j])
                self.update_params()

            print(f'Epoch={i} Loss: {log_loss(np.array(y).ravel(), np.array(predictions).ravel())}'
                  f' Accuracy: {accuracy_metric(np.array(y).ravel(), np.array(predictions).ravel())}'
                  f' --- Preds: {np.array(predictions).ravel()}')
