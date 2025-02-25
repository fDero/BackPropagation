from math import sqrt
from numpy.random import normal, uniform
from backpropy.expr import Expr, Parameter


class Neuron:
    def __init__(self, weights: list[Parameter], bias: Parameter):
        self.weights = weights
        self.bias = bias

    @staticmethod
    def default(inputs_count: int) -> "Neuron":
        weights = []
        for _ in range(inputs_count):
            random_weight = normal(0, sqrt(2 / inputs_count))
            weights.append(Parameter(random_weight))
        bias = Parameter(uniform(-1, 1))
        return Neuron(weights, bias)

    def eval(self, inputs: list[Expr]) -> Expr:
        assert len(inputs) == len(self.weights)
        output = self.bias
        for input_value, weight in zip(inputs, self.weights):
            output += input_value * weight
        return output.relu()