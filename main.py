import random

import numpy as np
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from MultilayerPerceptron.MLP import MLP
from MultilayerPerceptron.Layer import Layer

x_train = []
x_test = []
y_train = []
y_test = []


def classify(net_output):
    if net_output >= 0.5:
        return 1

    return 0


if __name__ == '__main__':
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
    #         y_test.append([sample_class])
    #     else:
    #         x_train.append(sample)
    #         y_train.append([sample_class])
    #
    # print(f'Len train: {len(x_train)}')
    # print(f'Len test: {len(x_test)}')

    x_train = [[0, 0], [1, 0], [0, 1], [1, 1]]
    y_train = [[0], [1], [1], [0]]

    net = MLP(lr=0.2, classify_function=classify)
    hidden_layer_1 = Layer(input_dim=len(x_train[0]), neurons=2, activation_function="sigmoid")
    output_layer = Layer(input_dim=2, neurons=len(y_train[0]), activation_function="sigmoid")

    net.add_layer(hidden_layer_1)
    net.add_layer(output_layer)

    net.optimize(x_train, y_train, epochs=1000)

    """
    Generate plot of input data and decision boundary.
    Ploting code from: 
    https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7
    """
    # setting plot properties like size, theme and axis limits
    sns.set_style('darkgrid')
    plt.figure(figsize=(20, 20))

    plt.axis('scaled')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    colors = {
        0: "ro",
        1: "go"
    }

    # plotting the four datapoints
    for i in range(len(x_train)):
        plt.plot([x_train[i][0]],
                 [x_train[i][1]],
                 colors[y_train[i][0]],
                 markersize=20)

    x_range = np.arange(-0.1, 1.1, 0.01)
    y_range = np.arange(-0.1, 1.1, 0.01)

    # creating a mesh to plot decision boundary
    xx, yy = np.meshgrid(x_range, y_range, indexing='ij')
    Z = np.array([[net.predict([x, y]) for x in x_range] for y in y_range])

    # using the contourf function to create the plot
    plt.contourf(xx, yy, Z, colors=['red', 'green', 'green', 'blue'], alpha=0.4)
    plt.show()