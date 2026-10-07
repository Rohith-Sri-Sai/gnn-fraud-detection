import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split
import os
from src.utils import load_params


def build_tfinance(path, train_ratio, val_ratio, split_seed) -> HeteroData:
    """
    T-Finance (Tang et al., ICML 2022): one node type (accounts on a financial platform),
    10 features, labels anomalous (fraud, money laundering, gambling) vs normal.
    There are no timestamps, so the benchmark protocol is a stratified random split.
    `path` is the plain-tensor export of the authors' DGL graph (see src/convert_tfinance.py).
    """
    print("1. Loading T-Finance...")
    raw = torch.load(path)
    y = raw["label"].argmax(1).long()   # labels are stored one-hot
    n = len(y)

    print("\n2. Splitting accounts (stratified random split)...")
    idx = np.arange(n)
    train_idx, rest = train_test_split(idx, stratify=y, train_size=train_ratio,
                                       random_state=split_seed, shuffle=True)
    # The remainder is divided between val and test in proportion val_ratio : (1 - train - val)
    val_share = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(rest, stratify=y[rest], train_size=val_share,
                                         random_state=split_seed, shuffle=True)

    print("\n3. Scaling features (train accounts only)...")
    # All 10 features are non-negative counts/amounts with heavy tails -> log1p, then
    # standardize with statistics of the training accounts only.
    x = torch.log1p(raw["x"].double().clamp_min(0))
    mean, std = x[train_idx].mean(0), x[train_idx].std(0).clamp_min(1e-6)
    x = ((x - mean) / std).float()

    print("\n4. Building graph...")
    data = HeteroData()
    data.target_node_type = "account"
    data["account"].x = x
    data["account"].y = y
    data["account"].num_nodes = n
    for name, ids in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[ids] = True
        data["account"][f"{name}_mask"] = mask
        print(f"   {name:5s}: {len(ids):6d} accounts, anomaly rate {y[mask].float().mean():.3f}")
    # The edge list is already symmetric (every edge is stored in both directions)
    data["account", "transacts_with", "account"].edge_index = raw["edge_index"].long()
    print(f"\n Graph Ready! Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    return data


if __name__ == "__main__":
    params = load_params()["preprocess"]
    g = build_tfinance(params["data_path"], params["train_ratio"], params["val_ratio"], params["split_seed"])

    # Save the PyTorch Geometric HeteroData object to disk
    os.makedirs("data/processed", exist_ok=True)
    torch.save(g, "data/processed/graph.pt")
    print("Saved processed graph to data/processed/graph.pt")
