from backpropy.expr import Expr


class Parameter(Expr):
    def __init__(self, value: float):
        self._value = value

    def value(self) -> float:
        return self._value

    def grad(self) -> dict["Parameter", float]:
        return {self: 1.0}

    def increment(self, inc: float):
        self._value += inc
