from abc import abstractmethod
from collections import Counter
from math import tanh


class Expr:
    @abstractmethod
    def grad(self) -> dict["Parameter", float]:
        pass

    @abstractmethod
    def value(self) -> float:
        pass

    @staticmethod
    def _merge_gradients(glx, grx: dict["Parameter", float]) -> dict["Parameter", float]:
        result = dict(glx.items())
        for k, v in grx.items():
            result[k] = result.get(k, 0.0) + v
        return result

    def __add__(self, other: "Expr") -> "Expr":
        return Sum(self, other)

    def __mul__(self, other: "Expr") -> "Expr":
        return Mul(self, other)

    def tanh(self) -> "Expr":
        return Tanh(self)

    def relu(self) -> "Expr":
        return ReLU(self)


class Parameter(Expr):
    def __init__(self, value: float):
        self._value = value

    def __repr__(self):
        return f"param({self._value})"

    def value(self) -> float:
        return self._value

    def grad(self) -> dict["Parameter", float]:
        return {self: 1.0}

    def increment(self, inc: float):
        self._value += inc


class Immutable(Expr):
    def __init__(self, value: float):
        self._value = value

    def __repr__(self):
        return f"imm({self._value})"

    def value(self) -> float:
        return self._value

    def grad(self) -> dict[Parameter, float]:
        return {}


class Mul(Expr):
    def __init__(self, lx: Expr, rx: Expr):
        self.lx = lx
        self.rx = rx

    def __repr__(self):
        return f"mul({self.lx}, {self.rx})"

    def value(self) -> float:
        return self.lx.value() * self.rx.value()

    def grad(self) -> dict[Parameter, float]:
        rx_value = self.rx.value()
        lx_value = self.lx.value()
        return self._merge_gradients(
            {k: v * rx_value for k, v in self.lx.grad().items()},
            {k: v * lx_value for k, v in self.rx.grad().items()}
        )


class ReLU(Expr):
    def __init__(self, arg: Expr, alpha=0.01):
        self.arg = arg
        self.alpha = alpha

    def __repr__(self):
        return f"relu({self.arg})"

    def value(self) -> float:
        return max(0.0, self.arg.value())

    def grad(self) -> dict["Parameter", float]:
        arg_value = self.arg.value()
        grad_factor = 1.0 if arg_value > 0 else self.alpha
        return {k: v * grad_factor for k, v in self.arg.grad().items()}


class Sum(Expr):
    def __init__(self, lx: Expr, rx: Expr):
        self.lx = lx
        self.rx = rx

    def __repr__(self):
        return f"sum({self.lx}, {self.rx})"

    def value(self) -> float:
        return self.lx.value() + self.rx.value()

    def grad(self) -> dict[Parameter, float]:
        return self._merge_gradients(
            {k: v for k, v in self.lx.grad().items()},
            {k: v for k, v in self.rx.grad().items()}
        )


class Tanh(Expr):
    def __init__(self, arg: Expr):
        self.arg = arg

    def __repr__(self):
        return f"tanh({self.arg})"

    def value(self) -> float:
        return tanh(self.arg.value())

    def grad(self) -> dict[Parameter, float]:
        tanh_value = self.value()
        tanh_grad_factor = 1 - tanh_value ** 2
        return {k: v * tanh_grad_factor for k, v in self.arg.grad().items()}
