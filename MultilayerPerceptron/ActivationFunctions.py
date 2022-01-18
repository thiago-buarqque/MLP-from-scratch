import math
import numpy


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    return numpy.maximum(x, 0)
