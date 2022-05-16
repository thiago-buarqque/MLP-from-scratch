import random
import numpy as np
import pandas as pd

from MultilayerPerceptron.ActivationFunctions import sigmoid, relu
from MultilayerPerceptron.MLP import MLP
from Perceptron.Perceptron import Perceptron
from MultilayerPerceptron.Layer import Layer

x_train = []
x_test = []
y_train = []
y_test = []

if __name__ == '__main__':
    # print(f'{[[[1]],2,3,4,5, 6][1:6]}')
    # hidden_layer_1 = Layer(input_dim=2, neurons=2, func="relu")

    net = MLP()
    hidden_layer_1 = Layer(input_dim=2, neurons=2, func="sigmoid",
                           initial_weights=None,
                           initial_biases=None)
    # hidden_layer_2 = Layer(input_dim=2, neurons=3, func="tanh")
    output_layer = Layer(input_dim=2, neurons=1, func="sigmoid",
                         initial_weights=None,
                         initial_biases=None)

    net.add_layer(hidden_layer_1)
    # net.add_layer(hidden_layer_2)
    net.add_layer(output_layer)

    net.optimize([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]], 50)

    # data = pd.read_csv("./sonar.all-data.csv", header=None)
    # data = data.sample(frac=1)
    #
    # for d in data.iterrows():
    #     sample = []
    #     for i in range(len(d[1]) - 1):
    #         sample.append(float(d[1][i]))
    #
    #     sample_class = 1 if d[1][len(d[1]) - 1] == 'M' else 0
    #     if random.random() > 0.5 and len(x_test) < 41:
    #         x_test.append(sample)
    #         y_test.append(sample_class)
    #     else:
    #         x_train.append(sample)
    #         y_train.append(sample_class)
    #
    # print(f'Len train: {len(x_train)}')
    # print(f'Len test: {len(x_test)}')
    #
    # perceptron = Perceptron(input_dim=len(x_train[0]), lr=0.01)
    # perceptron.train(200, x_train, y_train)
    #
    # perceptron.test(x_test, y_test)
