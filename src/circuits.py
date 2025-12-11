import pennylane as qml
from pennylane import numpy as np


def actor_qnode(n_wires: int = 10, layers: int = 2):
    dev = qml.device("default.qubit", wires=n_wires)
    weight_shapes = {"weights": (layers, n_wires, 3)}

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_wires))
        qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]  # 4 logits (actions)

    return circuit, weight_shapes


def critic_qnode(n_wires: int = 10, layers: int = 2):
    dev = qml.device("default.qubit", wires=n_wires)
    weight_shapes = {"weights": (layers, n_wires, 3)}

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_wires))
        qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
        return qml.expval(qml.PauliZ(0))

    return circuit, weight_shapes
