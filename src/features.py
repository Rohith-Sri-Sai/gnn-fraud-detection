import torch


def with_label_features(x, y, known):
    """Label trick inputs: append [is_known, is_fraud] to the node features.
    `known` marks the nodes whose label may be used; every other node gets [0, 0].
    Shared by training (src/trainer.py) and scoring (src/predict.py) so both build
    exactly the same model inputs."""
    known = known.float()
    label_feat = torch.stack([known, (y == 1).float() * known], dim=1)
    return torch.cat([x, label_feat], dim=1)
