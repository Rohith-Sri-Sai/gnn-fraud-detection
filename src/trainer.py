import torch
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import numpy as np
import copy
from src.model import GraphModel, EnsembleModel
from src.features import with_label_features
import os
import json
from src.utils import load_params
import mlflow
import mlflow.pytorch
import subprocess

# Defaults for every optional train param
DEFAULTS = dict(
    lr=0.001, weight_decay=0.0, num_layers=2, dropout=0.3, residual=True, norm=True, aggr="mean",
    pos_weight_power=1.0, scheduler="none", seed=42, log_every=1,
    label_trick=False, label_rate=0.5,
)


def test_metrics(probs, targets):
    # PR-AUC is better for imbalance
    return {"test_roc_auc": roc_auc_score(targets, probs),
            "test_pr_auc": average_precision_score(targets, probs)}


def pick_alert_threshold(probs, targets, target_precision):
    """Lowest score threshold whose precision reaches `target_precision` (i.e. the most
    recall at that precision). Chosen on VALIDATION data. Falls back to the best-F1
    threshold if the target precision is never reached."""
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    precision, recall = precision[:-1], recall[:-1]   # last point has no threshold
    ok = np.where(precision >= target_precision)[0]
    if len(ok):
        return float(thresholds[ok[0]]), "target_precision"
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1)]), "best_f1_fallback"


def alert_metrics(probs, targets, threshold, prefix):
    alert = probs >= threshold
    tp = int((alert & (targets == 1)).sum())
    return {f"{prefix}_alert_precision": tp / max(int(alert.sum()), 1),
            f"{prefix}_alert_recall": tp / max(int((targets == 1).sum()), 1),
            f"{prefix}_num_alerts": int(alert.sum())}


class GraphTrainer:
    """
    Full-batch training: the whole graph sits on the GPU as sparse adjacency matrices and
    every epoch is one gradient step on all training nodes. T-Finance is small (39k accounts)
    but dense (~1,000 neighbors per account), so this is faster than sampling subgraphs.

    label_trick=True (Wang et al. 2021, "Bag of Tricks for Node Classification"):
      the known TRAINING labels are appended to the target-node features as [is_known, is_fraud],
      so message passing can use the labels of a node's neighbors. Each epoch a random
      `label_rate` share of training nodes exposes its label and the loss is computed only on
      the remaining training nodes, so a node never sees its own label. At evaluation every
      training label is exposed; validation and test labels are never used as inputs.
    """
    def __init__(self, data, hidden_channel=128, epochs=300, metric_prefix="", **kwargs):
        unknown = set(kwargs) - set(DEFAULTS)
        assert not unknown, f"unknown train params: {unknown}"
        self.cfg = {**DEFAULTS, **kwargs}
        cfg = self.cfg
        self.metric_prefix = metric_prefix
        self.epochs = epochs
        self.tt = data.target_node_type   # node type being classified

        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {self.device}")

        # 1. Initialize Model
        self.model = GraphModel(
            hidden_channels=hidden_channel,
            out_channels=2,
            metadata=data.metadata(),
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            residual=cfg["residual"],
            norm=cfg["norm"],
            target_node_type=self.tt,
            aggr=cfg["aggr"],
        ).to(self.device)

        # 2. Move the graph to the GPU as SparseTensor adjacencies: SAGEConv then aggregates
        # with a sparse matmul instead of materializing one message per edge (42M edges).
        self.full = T.ToSparseTensor()(copy.copy(data)).to(self.device)
        self.store = self.full[self.tt]
        with torch.no_grad():
            self._forward()   # materializes the lazy (-1) layer sizes

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.scheduler = None
        if cfg["scheduler"] == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # 3. Handle the class imbalance
        # computed from the actual training labels rather than hardcoded, so it always reflects the real ratio.
        train_labels = self.store.y[self.store.train_mask]
        num_pos = int((train_labels == 1).sum())
        num_neg = int((train_labels == 0).sum())
        pos_weight = (num_neg / max(num_pos, 1)) ** cfg["pos_weight_power"]
        print(f"Class balance -> neg: {num_neg}, pos: {num_pos}, pos_weight: {pos_weight:.2f}")

        weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        self.best_val_pr = -1.0
        self.best_state_dict = None
        self.best_epoch = -1

    def _forward(self, known=None):
        """Forward pass on the whole graph. `known` marks the target nodes whose label is
        exposed as an input (label trick); default: every training node."""
        x_dict = dict(self.full.x_dict)
        if self.cfg["label_trick"]:
            known = self.store.train_mask if known is None else known
            x_dict[self.tt] = with_label_features(x_dict[self.tt], self.store.y, known)
        return self.model(x_dict, self.full.adj_t_dict)

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        known, mask = None, self.store.train_mask
        if self.cfg["label_trick"]:
            # Split the training nodes: one part exposes its labels, the other is scored
            expose = torch.rand(self.store.num_nodes, device=self.device) < self.cfg["label_rate"]
            known, mask = self.store.train_mask & expose, self.store.train_mask & ~expose

        loss = self.loss_fn(self._forward(known)[mask], self.store.y[mask])
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    @torch.no_grad() # Turn off gradients to save memory during evaluation
    def predict(self, split):
        """Fraud probability and label for every target node of a split ("train", "val" or "test")."""
        self.model.eval()
        mask = self.store[f"{split}_mask"]
        # Softmax converts raw output numbers into probabilities (0.0 to 1.0)
        probs = torch.softmax(self._forward()[mask], dim=1)[:, 1]
        return probs.cpu().numpy(), self.store.y[mask].cpu().numpy()

    def evaluate(self, split):
        probs, targets = self.predict(split)
        return roc_auc_score(targets, probs), average_precision_score(targets, probs)

    def run(self):
        print("\n Starting Training Loop...")

        for epoch in range(1, self.epochs + 1):
            loss = self.train_epoch()
            val_roc, val_pr = self.evaluate("val")

            is_best = val_pr > self.best_val_pr
            if is_best:
                self.best_val_pr = val_pr
                self.best_epoch = epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())

            if epoch % self.cfg["log_every"] == 0 or epoch == self.epochs:
                mlflow.log_metrics({f"{self.metric_prefix}train_loss": loss,
                                    f"{self.metric_prefix}val_roc_auc": val_roc,
                                    f"{self.metric_prefix}val_pr_auc": val_pr}, step=epoch)
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train Loss: {loss:.4f} | "
                    f"Val ROC-AUC: {val_roc:.4f} | "
                    f"Val PR-AUC: {val_pr:.4f}"
                    f"{'  <- best so far' if is_best else ''}"
                )

        # Restore the best checkpoint (by val PR-AUC) before final test eval,
        # instead of silently using whatever the last epoch happened to be.
        print(f"\nRestoring best checkpoint from epoch {self.best_epoch} "
              f"(val PR-AUC: {self.best_val_pr:.4f})")
        self.model.load_state_dict(self.best_state_dict)
        mlflow.log_param(f"{self.metric_prefix}best_epoch", self.best_epoch)
        mlflow.log_metric(f"{self.metric_prefix}best_val_pr_auc", self.best_val_pr)

        print("\n" + "=" * 50)
        print(" FINAL PRODUCTION TEST EVALUATION ")
        print("=" * 50)

        results = test_metrics(*self.predict("test"))
        for k, v in results.items():
            print(f"Final {k}: {v:.4f}")
        return results

def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    all_params = load_params()
    params = dict(all_params["train"])
    # Overridable via env so the same code works on the host or inside a container
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "Fraud Detection GNN - tfinance"))
    registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "gnn-fraud-tfinance-graphsage")

    # 1. Load the processed graph
    data = torch.load("data/processed/graph.pt",weights_only=False)
    store = data[data.target_node_type]

    # 2. Initialize and run trainer
    with mlflow.start_run():

        ensemble_size = params.pop("ensemble_size", 1)
        base_seed = params.get("seed", DEFAULTS["seed"])
        mlflow.log_params({**params, "ensemble_size": ensemble_size, "dataset": "tfinance"})
        mlflow.log_params({f"preprocess_{k}": v for k, v in all_params["preprocess"].items()})
        mlflow.log_params({
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
            "num_train": int(store.train_mask.sum()),
            "num_val": int(store.val_mask.sum()),
            "num_test": int(store.test_mask.sum()),
        })

        # Train every ensemble member (different seed) and keep its best-val checkpoint
        trainers, member_probs, member_val_probs = [], [], []
        for i in range(ensemble_size):
            print(f"\n##### Ensemble member {i + 1}/{ensemble_size} (seed {base_seed + i}) #####")
            member = GraphTrainer(data=data, **{**params, "seed": base_seed + i},
                                  metric_prefix=f"m{i}_" if ensemble_size > 1 else "")
            member_results = member.run()
            trainers.append(member)
            member_probs.append(member.predict("test"))
            member_val_probs.append(member.predict("val"))
            if ensemble_size > 1:
                mlflow.log_metrics({f"m{i}_{k}": v for k, v in member_results.items()})

        trainer = trainers[0]
        mlflow.set_tags({
            "dataset": "tfinance",
            "model_type": "GraphSAGE" + (f" x{ensemble_size} ensemble" if ensemble_size > 1 else ""),
            "framework": "pytorch_geometric",
            "device": str(trainer.device),
            "gpu_name": torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else "none",
            "torch_version": torch.__version__,
            "cuda_version": str(torch.version.cuda),
            "git_commit": _git_commit(),
        })

        if ensemble_size > 1:
            model = EnsembleModel([t.model for t in trainers])
            probs = np.mean([p for p, _ in member_probs], axis=0)
            results = test_metrics(probs, member_probs[0][1])
            print("\n" + "=" * 50 + f"\n ENSEMBLE OF {ensemble_size} TEST EVALUATION \n" + "=" * 50)
            for k, v in results.items():
                print(f"Final ensemble {k}: {v:.4f}")
        else:
            model = trainer.model
            results = member_results

        # Validation PR-AUC of the final (ensemble) model: used by the promotion stage,
        # so comparing model versions never reuses the test set.
        val_probs = np.mean([p for p, _ in member_val_probs], axis=0)
        val_targets = member_val_probs[0][1]
        results["val_pr_auc_final"] = average_precision_score(val_targets, val_probs)

        # Alert threshold: chosen on validation, then reported on test
        target_precision = all_params["alerts"]["target_precision"]
        threshold, rule = pick_alert_threshold(val_probs, val_targets, target_precision)
        test_probs = np.mean([p for p, _ in member_probs], axis=0)
        results.update(alert_metrics(val_probs, val_targets, threshold, "val"))
        results.update(alert_metrics(test_probs, member_probs[0][1], threshold, "test"))
        results["alert_threshold"] = threshold
        mlflow.log_params({"alert_target_precision": target_precision, "alert_threshold_rule": rule})
        print(f"\nAlert threshold {threshold:.4f} ({rule}) -> "
              f"test precision {results['test_alert_precision']:.3f}, "
              f"recall {results['test_alert_recall']:.3f}, {results['test_num_alerts']} alerts")

        # 3. Save model weights
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/graph_sage_model.pt")
        # Log a CPU copy so the model can be loaded on machines without a GPU;
        # registered_model_name creates a new version in the Model Registry on every run.
        # 'pickle' because the 'pt2' traced format can't take the x_dict/adjacency dict inputs.
        model_info = mlflow.pytorch.log_model(
            copy.deepcopy(model).cpu(),
            name="model",
            serialization_format="pickle",
            code_paths=["src"],  # pickled model references src.model.GraphModel
            registered_model_name=registered_model_name
        )

        # 4. Evaluate and save metrics
        for k, v in results.items():
            mlflow.log_metric(k, v)
        metrics = {k: float(v) for k, v in results.items()}

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact("params.yaml")

        # 5. Attach what serving needs to the model version itself, and record which
        # version this run created for the promotion stage (src/promote.py)
        version = str(model_info.registered_model_version)
        client = mlflow.MlflowClient()
        for key in ("alert_threshold", "val_pr_auc_final", "test_pr_auc"):
            client.set_model_version_tag(registered_model_name, version, key, f"{results[key]:.6f}")
        client.set_model_version_tag(registered_model_name, version, "label_trick", str(params.get("label_trick", False)))
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/registered_model.json", "w") as f:
            json.dump({"name": registered_model_name, "version": version,
                       "run_id": mlflow.active_run().info.run_id,
                       **{k: metrics[k] for k in ("val_pr_auc_final", "test_pr_auc", "alert_threshold")}},
                      f, indent=4)
    print(f"Saved model weights and metrics.json; registered {registered_model_name} v{version}")
