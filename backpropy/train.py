from backpropy import Network, Immutable, Expr


class Trainer:
    def __init__(self, network: Network, training_inputs, expected_outputs: list[list[Immutable]]):
        self.network = network
        self.training_inputs = training_inputs
        self.expected_outputs = expected_outputs

    def _compute_diff(self) -> Expr:
        total_diff = Immutable(0.0)
        for training_input, expected in zip(self.training_inputs, self.expected_outputs):
            result = self.network.eval(training_input)
            for result_elem, expected_elem in zip(result, expected):
                total_diff += expected_elem + (result_elem * Immutable(-1.0))
        return total_diff

    def _compute_loss(self) -> Expr:
        diff = self._compute_diff()
        return diff * diff

    def train(self, learning_rate: float = 0.0001) -> "TrainingResults":
        assert learning_rate > 0.0
        loss = self._compute_loss()
        print(loss.grad(), loss.value(), loss)
        for param, grad in loss.grad().items():
            print(param, grad)
            param.increment(grad * learning_rate * -1)
            print(param, grad)
        return TrainingResults(loss.value())


class TrainingResults:
    def __init__(self, loss: float):
        self.loss = loss

    def __repr__(self):
        return f"Loss: {self.loss}"
