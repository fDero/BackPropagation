from backpropy.net import Neuron
from backpropy.expr import Expr


class Layer:
    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

    @staticmethod
    def default(inputs_count: int, neurons_count: int) -> "Layer":
        return Layer([Neuron.default(inputs_count) for _ in range(neurons_count)])

    def eval(self, inputs: list[Expr]) -> list[Expr]:
        return [neuron.eval(inputs) for neuron in self.neurons]
