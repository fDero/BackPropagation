from backpropy.expr import Expr, Parameter


class Mul(Expr):
    def __init__(self, lx: Expr, rx: Expr):
        self.lx = lx
        self.rx = rx

    def value(self) -> float:
        return self.lx.value() * self.rx.value()

    def grad(self) -> dict[Parameter, float]:
        rx_value = self.rx.value()
        lx_value = self.lx.value()
        return self._merge_gradients(
            {k: v * rx_value for k, v in self.lx.grad().items()},
            {k: v * lx_value for k, v in self.rx.grad().items()}
        )
