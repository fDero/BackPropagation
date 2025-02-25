from backpropagation.expr import Expr, Parameter


class ReLU(Expr):
    def __init__(self, arg: Expr, alpha=0.01):
        self.arg = arg
        self.alpha = alpha

    def value(self) -> float:
        return max(0.0, self.arg.value())

    def grad(self) -> dict["Parameter", float]:
        arg_value = self.arg.value()
        grad_factor = 1.0 if arg_value > 0 else self.alpha
        return {k: v * grad_factor for k, v in self.arg.grad().items()}
