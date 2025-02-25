from backpropagation.net import Layer
from backpropagation.expr import Expr


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
