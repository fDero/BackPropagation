from backpropy.expr import Expr, Parameter


class Sum(Expr):
    def __init__(self, lx: Expr, rx: Expr):
        self.lx = lx
        self.rx = rx

    def value(self) -> float:
        return self.lx.value() + self.rx.value()

    def grad(self) -> dict[Parameter, float]:
        return self._merge_gradients(
            {k: v for k, v in self.lx.grad().items()},
            {k: v for k, v in self.rx.grad().items()}
        )
