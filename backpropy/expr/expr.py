from backpropy.expr import Parameter, Sum, Mul, Tanh, ReLU
from abc import abstractmethod
from collections import Counter


class Expr:
    @abstractmethod
    def grad(self) -> dict["Parameter", float]:
        pass

    @abstractmethod
    def value(self) -> float:
        pass

    @staticmethod
    def _merge_gradients(glx, grx: dict["Parameter", float]) -> dict["Parameter", float]:
        return dict(Counter(glx) + Counter(grx))

    def __add__(self, other: "Expr") -> "Expr":
        return Sum(self, other)

    def __mul__(self, other: "Expr") -> "Expr":
        return Mul(self, other)

    def tanh(self) -> "Expr":
        return Tanh(self)

    def relu(self) -> "Expr":
        return ReLU(self)
