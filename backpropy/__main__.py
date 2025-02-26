from backpropy import *

from random import uniform

training_data_1 = [
    [
        Immutable(uniform(-1, 1)),
        Immutable(uniform(-1, 1)),
        Immutable(uniform(-1, 1)),
        Immutable(uniform(-1, 1)),
    ],
    [
        Immutable(uniform(-1, 1))
    ]
]

training_data_2 = [
    [
        Immutable(uniform(-1, 1)),
        Immutable(uniform(-1, 1)),
        Immutable(uniform(-1, 1)),
        Immutable(uniform(-1, 1)),
    ],
    [
        Immutable(uniform(-1, 1)),
    ]
]

training_data_3 = [
    [
        Immutable(uniform(-1, 1)),
        Immutable(uniform(-1, 1)),
        Immutable(uniform(-1, 1)),
        Immutable(uniform(-1, 1)),
    ],
    [
        Immutable(uniform(-1, 1)),
    ]
]

training_data_4 = [
    [
        Immutable(uniform(-1, 1)),
        Immutable(uniform(-1, 1)),
        Immutable(uniform(-1, 1)),
        Immutable(uniform(-1, 1)),
    ],
    [
        Immutable(uniform(-1, 1)),
    ]
]

if __name__ == "__main__":
    neural_network = Network.default(4, [5, 5], 1)
    learning_rate = 0.0001

    velocities = {}
    momentum = 0.7
    total_abs_grad = 0.0

    for i in range(100000):
        loss = Immutable(0.0)
        for training_data in [training_data_1, training_data_2, training_data_3, training_data_4]:
            out = neural_network.eval(training_data[0])

            neg_target = Immutable(-1) * training_data[1][0]
            diff = out[0] + neg_target
            loss = loss + (diff * diff)


        if i % 10 == 0 and i > 10:
            print(f"Epoch {i}, Total Loss: {loss.value():.8f}")

        for param, grad in loss.grad().items():
            total_abs_grad += abs(grad)
            if param not in velocities:
                velocities[param] = 0.0

            velocities[param] = momentum * velocities[param] - learning_rate * grad
            param.increment(velocities[param])

        if i % 10 == 0 and i > 10:
            print(f"total abs grad: {total_abs_grad}")

        total_abs_grad = 0.0