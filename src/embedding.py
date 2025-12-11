from typing import List
import torch
import math


def encode_state(history: List[int], n_wires: int = 10) -> torch.Tensor:
    """
    Encode a 9-length action history (5 atoms + 4 bonds) of ints in [0,3] into
    rotation angles for AngleEmbedding. Scales token id to [0, pi], pads to n_wires.
    """
    angles = [ (math.pi / 3.0) * x for x in history ]  # map 0..3 -> 0..pi
    if len(angles) < n_wires:
        angles += [0.0] * (n_wires - len(angles))
    else:
        angles = angles[:n_wires]
    return torch.tensor(angles, dtype=torch.float32)
