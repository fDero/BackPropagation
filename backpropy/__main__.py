from backpropy import *

from random import uniform

training_data = [
    [ Immutable(uniform(-1, 1)), Immutable(uniform(-1, 1)), Immutable(uniform(-1, 1)), Immutable(uniform(-1, 1)), ],
    [ Immutable(uniform(-1, 1)), Immutable(uniform(-1, 1)), Immutable(uniform(-1, 1)), Immutable(uniform(-1, 1)), ],
    [ Immutable(uniform(-1, 1)), Immutable(uniform(-1, 1)), Immutable(uniform(-1, 1)), Immutable(uniform(-1, 1)), ],
    [ Immutable(uniform(-1, 1)), Immutable(uniform(-1, 1)), Immutable(uniform(-1, 1)), Immutable(uniform(-1, 1)), ],
]

expected_outputs = [
    [ Immutable(uniform(-1, 1)) ],
    [ Immutable(uniform(-1, 1)) ],
    [ Immutable(uniform(-1, 1)) ],
    [ Immutable(uniform(-1, 1)) ],
]

if __name__ == "__main__":
    network = Network.default(4, [5, 5], 1)
    trainer = Trainer(network, training_data, expected_outputs)
    for i in range(100):
        print(trainer.train())