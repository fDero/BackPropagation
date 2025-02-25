from backpropy.expr import Expr, Parameter


class Immutable(Expr):
    def __init__(self, value: float):
        self._value = value

    def value(self) -> float:
        return self._value

    def grad(self) -> dict[Parameter, float]:
        return {}
