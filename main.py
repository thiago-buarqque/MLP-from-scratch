import math

import numpy as np
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from MultilayerPerceptron.MLP import MLP
from MultilayerPerceptron.Layer import Layer


def classify(net_output):
    if net_output >= 0.5:
        return 1

    return 0


def handle_nan_values(dataset, current_row_counter, new_row, backup_column):
    for i, row_column in enumerate(new_row):
        if math.isnan(row_column):
            sub_range = current_row_counter if i != len(
                new_row) - 1 else current_row_counter + len(new_row) - 1

            for j in reversed(range(sub_range)):
                if not math.isnan(float(dataset.iloc[[j]][backup_column])):
                    new_row[i] = float(dataset.iloc[[j]][backup_column])
                    break


def generate_dataset(dataset, desired_column, backup_column, period=5):
    j = period - 1

    result = []
    while (j + period) <= len(dataset) - 1:
        new_row = []
        for i in reversed(range(period)):
            new_row.append(float(dataset.iloc[[j - i]][desired_column]))

        new_row.append(float(dataset.iloc[[j + period]][desired_column]))

        handle_nan_values(dataset, j, new_row, backup_column)

        result.append(new_row)
        j += 1

    columns = []
    for i in range(period):
        if period - i - 1 == 0:
            columns.append("Dia(k)")
        else:
            columns.append(f"Dia(k-{period - i - 1})")

    columns.append("DiaObj")

    return pd.DataFrame(result, columns=columns)


if __name__ == '__main__':
    acoes = pd.read_csv('./acoes_bb_2017_2022.csv')

    period = 5
    acoes = generate_dataset(acoes, desired_column="Open",
                             backup_column="Close", period=period)

    # Y = acoes.pop("DiaObj")
    # X = acoes

    real_world_data = acoes.tail(5)

    acoes.drop(acoes.tail(5).index, inplace=True)
    x_train = acoes.sample(frac=0.9, random_state=0)
    x_test = acoes.drop(x_train.index)

    y_train = x_train.pop('DiaObj')
    y_test = x_test.pop('DiaObj')

    real_labels = real_world_data.pop('DiaObj').values.tolist()
    real_world_data = real_world_data.values.tolist()

    # [ ] Remove random state
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=0)
    #
    x_train = x_train.values.tolist()
    x_test = x_test.values.tolist()
    y_train = y_train.values.tolist()
    y_test = y_test.values.tolist()

    net = MLP(lr=0.01)
    hidden_layer_1 = Layer(input_dim=len(x_train[0]), neurons=64, activation_function="relu")
    output_layer = Layer(input_dim=64, neurons=1, activation_function="relu")

    net.add_layer(hidden_layer_1)
    net.add_layer(output_layer)

    net.optimize(x_train, y_train, epochs=100)

    preds = []
    for sample in real_world_data:
        preds.append(net.evaluate(sample))

    print(f"\nPredictions: {list(preds)}")
    print(f"Real: {list(real_labels)}")

    """
    Generate plot of input data and decision boundary.
    Ploting code from: 
    https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7
    """
    # setting plot properties like size, theme and axis limits
    # sns.set_style('darkgrid')
    # plt.figure(figsize=(20, 20))
    #
    # plt.axis('scaled')
    # plt.xlim(-0.1, 1.1)
    # plt.ylim(-0.1, 1.1)
    #
    # colors = {
    #     0: "ro",
    #     1: "go"
    # }
    #
    # # plotting the four datapoints
    # for i in range(len(x_train)):
    #     plt.plot([x_train[i][0]],
    #              [x_train[i][1]],
    #              colors[y_train[i][0]],
    #              markersize=20)
    #
    # x_range = np.arange(-0.1, 1.1, 0.01)
    # y_range = np.arange(-0.1, 1.1, 0.01)
    #
    # # creating a mesh to plot decision boundary
    # xx, yy = np.meshgrid(x_range, y_range, indexing='ij')
    # Z = np.array([[net.predict([x, y]) for x in x_range] for y in y_range])
    #
    # # using the contourf function to create the plot
    # plt.contourf(xx, yy, Z, colors=['red', 'green', 'green', 'blue'], alpha=0.4)
    # plt.show()
