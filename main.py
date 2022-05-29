import pandas as pd
import random

from MultilayerPerceptron.MLP import MLP
from MultilayerPerceptron.Layer import Layer

x_train = []
x_test = []
y_train = []
y_test = []

if __name__ == '__main__':
    data = pd.read_csv("./sonar.all-data.csv", header=None)
    data = data.sample(frac=1)

    for d in data.iterrows():
        sample = []
        for i in range(len(d[1]) - 1):
            sample.append(float(d[1][i]))

        sample_class = 1 if d[1][len(d[1]) - 1] == 'M' else 0
        if random.random() > 0.5 and len(x_test) < 41:
            x_test.append(sample)
            y_test.append([sample_class])
        else:
            x_train.append(sample)
            y_train.append([sample_class])

    print(f'Len train: {len(x_train)}')
    print(f'Len test: {len(x_test)}')

    # x_train = [[0, 0], [1, 0], [0, 1], [1, 1]]
    # y_train = [[0], [1], [1], [0]]

    net = MLP(lr=0.01)
    hidden_layer_1 = Layer(input_dim=len(x_train[0]), neurons=64, func="sigmoid")
    output_layer = Layer(input_dim=64, neurons=len(y_train[0]), func="sigmoid")

    net.add_layer(hidden_layer_1)
    net.add_layer(output_layer)

    net.optimize(x_train, y_train, epochs=500)
