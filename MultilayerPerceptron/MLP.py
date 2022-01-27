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
        self.lr = 0.05

    def add_layer(self, layer):
        if len(self.layers) == 0:
            self.input_dim = layer.input_dim
        self.layers.append(layer)

        for i in range(len(self.layers) - 1):
            if self.layers[i].next_layer is None:
                self.layers[i].set_next_layer(self.layers[i + 1])
                self.layers[i].layer_name = f'{i}'

    def get_layers(self):
        print(f'Printando {len(self.layers)} camadas')
        for layer in self.layers:
            print(layer)

    def backward_propagate_error(self, expected_output):
        for i in range(len(self.layers) - 1, -1, -1):
            curr_layer = self.layers[i]

            layer_errors = list()
            layer_weights = curr_layer.get_weights()

            # It's a hidden layer
            if i != len(self.layers) - 1:
                for j, neuron_weights in enumerate(layer_weights):
                    error = 0.0

                    next_layer_weights = self.layers[i + 1].get_weights()
                    next_layer_deltas = self.layers[i + 1].deltas
                    for k, next_layer_neuron in enumerate(next_layer_weights):
                        error += next_layer_neuron[j] * next_layer_deltas[k]

                    layer_errors.append(error)
            else:
                for j, out_neuron in enumerate(curr_layer.layer_output):
                    layer_errors.append(out_neuron - expected_output[j])

            for j, out_neuron in enumerate(curr_layer.layer_output):
                curr_layer.deltas.append(layer_errors[j] * curr_layer.activation_derivative(out_neuron))

    def update_params(self, net_input):
        for i, layer in enumerate(self.layers):
            inputs = net_input
            if i != 0:
                inputs = self.layers[i - 1].layer_output

            weights, biases = layer.get_weights_and_biases()

            # print(f'Weights before: {weights}')
            # Updating weights
            for j, neuron_weights in enumerate(weights):
                for k in range(len(inputs)):
                    neuron_weights[k] = neuron_weights[k] - (self.lr * layer.deltas[j] * inputs[k])
            # print(f'Weights after: {weights}')

            # Updating biases
            # print(f'Biases before: {biases}')
            for j in range(len(biases)):
                biases[j] = biases[j] - (self.lr * layer.deltas[j])
            # print(f'Biases after: {biases}')

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

                net_pred = 1 if next_layer_input_data > 0.5 else 0

                predictions.append(net_pred)
                # print(f'Pred for sample {j+1}: {next_layer_input_data}')

                self.backward_propagate_error(y[j])
                self.update_params(sample)

            print(f'Epoch={i} Loss: {log_loss(np.array(y).ravel(), np.array(predictions).ravel())}'
                  f' Accuracy: {accuracy_metric(np.array(y).ravel(), np.array(predictions).ravel())}'
                  f' --- Preds: {np.array(predictions).ravel()}')
