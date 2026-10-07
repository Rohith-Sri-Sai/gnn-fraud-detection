"""
Batch scoring: load the serving model (registry alias, default `champion`) and score
every account in the graph. Used as a CLI job and by the API (src/api.py).

The model is transductive: it scores accounts inside the graph and, with the label trick,
takes the known TRAINING labels as input features. Accounts with a known label are
reported but never raised as alerts - alerts are only for accounts we have no label for.

    python -m src.predict --output outputs/scores.csv
"""
import argparse
import csv
import os
import time
import torch
import torch_geometric.transforms as T
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from src.features import with_label_features

DEFAULT_MODEL_NAME = "gnn-fraud-tfinance-graphsage"


def score_accounts(graph_path="data/processed/graph.pt", model_name=None, alias=None):
    model_name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    alias = alias or os.getenv("MODEL_ALIAS", "champion")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    t0 = time.time()
    mv = MlflowClient().get_model_version_by_alias(model_name, alias)
    model = mlflow.pytorch.load_model(f"models:/{model_name}@{alias}", map_location="cpu").eval()

    data = torch.load(graph_path, weights_only=False)
    tt = data.target_node_type
    g = T.ToSparseTensor()(data)
    store = g[tt]
    x_dict = dict(g.x_dict)
    if mv.tags.get("label_trick", "True") == "True":
        x_dict[tt] = with_label_features(store.x, store.y, store.train_mask)   # training labels only

    with torch.no_grad():
        probs = torch.softmax(model(x_dict, g.adj_t_dict), dim=1)[:, 1].numpy()

    threshold = float(mv.tags["alert_threshold"])
    known = store.train_mask.numpy()
    labels = store.y.numpy()
    return {
        "model": {"name": model_name, "alias": alias, "version": mv.version, "run_id": mv.run_id,
                  "alert_threshold": threshold,
                  "val_pr_auc": float(mv.tags.get("val_pr_auc_final", "nan")),
                  "test_pr_auc": float(mv.tags.get("test_pr_auc", "nan"))},
        "probs": probs,
        "known_label": known,                      # True = label known (training account)
        "labels": labels,                          # only exposed for known accounts
        "alert": (probs >= threshold) & ~known,    # alerts only for unlabeled accounts
        "scored_in_s": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="data/processed/graph.pt")
    ap.add_argument("--output", default="outputs/scores.csv")
    args = ap.parse_args()

    s = score_accounts(args.graph)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["account_id", "fraud_probability", "alert", "known_label"])
        for i, p in enumerate(s["probs"]):
            known = "fraud" if s["known_label"][i] and s["labels"][i] == 1 else \
                    "normal" if s["known_label"][i] else ""
            w.writerow([i, f"{p:.6f}", int(s["alert"][i]), known])
    m = s["model"]
    print(f"Scored {len(s['probs'])} accounts with {m['name']}@{m['alias']} (v{m['version']}) "
          f"in {s['scored_in_s']}s; threshold {m['alert_threshold']:.4f}; "
          f"{int(s['alert'].sum())} alerts -> {args.output}")
