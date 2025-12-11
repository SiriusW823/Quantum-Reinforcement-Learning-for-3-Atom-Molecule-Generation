from __future__ import annotations

import torch
import pennylane as qml


def make_torch_pqc(n_wires: int, layers: int):
    """
    Build a TorchLayer PQC that maps input features -> expvals on each wire
    using AngleEmbedding + StronglyEntanglingLayers.
    """
    dev = qml.device("default.qubit", wires=n_wires)
    weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_wires)}

    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_wires))
        qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_wires)]

    return qml.qnn.TorchLayer(qnode, weight_shapes=weight_shapes, init_method=torch.nn.init.normal)
