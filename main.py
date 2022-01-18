import random
import numpy as np
import pandas as pd

from MultilayerPerceptron.ActivationFunctions import sigmoid, relu
from Perceptron.Perceptron import Perceptron
from MultilayerPerceptron.Layer import Layer

x_train = []
x_test = []
y_train = []
y_test = []

if __name__ == '__main__':
    l = Layer(input_dim=3, output_dim=2, next_layer=Layer(2, 2), func="relu")
    print(l)
    print(l.feed_layer([.25, .5, .9]))
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
