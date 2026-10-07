"""
One-time conversion of the T-Finance DGL graph into plain tensors, so the project itself
does not depend on DGL. Download `tfinance.zip` from the authors' Google Drive
(https://github.com/squareRoot3/Rethinking-Anomaly-Detection), unzip it into data/raw/tfinance/,
then run this in any environment that has dgl + torch installed:

    python -m src.convert_tfinance data/raw/tfinance/tfinance data/raw/tfinance/tfinance.pt
"""
import sys
import dgl
import torch

if __name__ == "__main__":
    src_path, dst_path = sys.argv[1], sys.argv[2]
    g = dgl.load_graphs(src_path)[0][0]
    src, dst = g.edges()
    torch.save({"edge_index": torch.stack([src, dst]).long(),
                "x": g.ndata["feature"].float(),
                "label": g.ndata["label"]}, dst_path)
    print(f"Saved {g.num_nodes()} nodes, {g.num_edges()} edges to {dst_path}")
