from math import sqrt
from numpy.random import normal, uniform
from backpropy import Parameter, Expr


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


class Layer:
    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

    @staticmethod
    def default(inputs_count: int, neurons_count: int) -> "Layer":
        return Layer([Neuron.default(inputs_count) for _ in range(neurons_count)])

    def eval(self, inputs: list[Expr]) -> list[Expr]:
        return [neuron.eval(inputs) for neuron in self.neurons]


class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    @staticmethod
    def default(inputs_count: int, hidden_layers_topology: list[int], outputs_count: int) -> "Network":
        complete_topology = [inputs_count] + hidden_layers_topology + [outputs_count]
        layers = []
        for i in range(len(complete_topology) - 1):
            layers.append(Layer.default(complete_topology[i], complete_topology[i + 1]))
        return Network(layers)

    def eval(self, inputs: list[Expr]) -> list[Expr]:
        data = inputs
        for layer in self.layers:
            data = layer.eval(data)
        return data
