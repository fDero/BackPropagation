import unittest
from backpropy import Expr, Parameter, Immutable

class TestExpr(unittest.TestCase):
    def test_gradient_simple(self):
        five = Parameter(5.0)
        six = Parameter(6.0)
        two = Parameter(2.0)
        expr = (five + two) * six
        grads = expr.grad()
        assert grads[five] == 6.0
        assert grads[six] == 7.0
        assert grads[two] == 6.0
        assert len(grads) == 3

    def test_gradient_with_one_immutable(self):
        five = Immutable(5.0)
        six = Parameter(6.0)
        two = Parameter(2.0)
        expr = (five + two) * six
        grads = expr.grad()
        assert grads[six] == 7.0
        assert grads[two] == 6.0
        assert len(grads) == 2

    def test_multiply_by_negative_one(self):
        param = Parameter(0.6)
        neg_one = Immutable(-1.0)
        expr = param * neg_one
        grads = expr.grad()
        assert len(grads) == 1

    def test_gradient_with_realistic_workload(self):
        inp = Immutable(0.7)
        weight1 = Parameter(-0.2)
        bias1 = Parameter(0.4)
        neuron_output = (inp * weight1 + bias1)
        expected = Immutable(0.4)
        diff = expected + (neuron_output * Immutable(-1.0))
        loss = diff * diff
        grads = loss.grad()
        neuron_grads = neuron_output.grad()
        assert len(grads) == len(neuron_grads)
        assert len(grads) > 0