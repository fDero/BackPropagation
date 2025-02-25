from backpropagation.expr import Expr, Parameter
from math import tanh


class Tanh(Expr):
    def __init__(self, arg: Expr):
        self.arg = arg

    def value(self) -> float:
        return tanh(self.arg.value())

    def grad(self) -> dict[Parameter, float]:
        tanh_value = self.value()
        tanh_grad_factor = 1 - tanh_value ** 2
        return {k: v * tanh_grad_factor for k, v in self.arg.grad().items()}
