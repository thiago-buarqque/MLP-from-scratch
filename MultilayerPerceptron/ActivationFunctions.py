import math
import numpy


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def relu(x):
    return numpy.maximum(x, 0)


def relu_derivative(x):
    return 0 if x < 0 else 1
