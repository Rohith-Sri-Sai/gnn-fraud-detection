"""
DVC stage 3: decide whether the model version the train stage just registered becomes
the serving model, by moving the MLflow registry alias (default `champion`) to it.

Rule: promote if
  1. its test PR-AUC is at least `promote.min_test_pr_auc` (a sanity floor), and
  2. there is no champion yet, or its VALIDATION PR-AUC is >= the champion's.
Comparing versions on validation (not test) keeps the test set out of model selection.
Previous versions are never deleted; only the alias moves.
"""
import json
import os
from datetime import datetime, timezone
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from src.utils import load_params

if __name__ == "__main__":
    params = load_params()["promote"]
    alias = params["alias"]
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()

    with open("outputs/registered_model.json") as f:
        new = json.load(f)
    name, version = new["name"], new["version"]

    try:
        champ = client.get_model_version_by_alias(name, alias)
        champ_val = float(champ.tags["val_pr_auc_final"]) if "val_pr_auc_final" in champ.tags else None
    except MlflowException:
        champ, champ_val = None, None

    if new["test_pr_auc"] < params["min_test_pr_auc"]:
        promoted, reason = False, f"test PR-AUC {new['test_pr_auc']:.4f} < floor {params['min_test_pr_auc']}"
    elif champ is None:
        promoted, reason = True, "no current champion"
    elif champ.version == version:
        promoted, reason = True, "already champion"
    elif champ_val is None or new["val_pr_auc_final"] >= champ_val:
        promoted, reason = True, (f"val PR-AUC {new['val_pr_auc_final']:.4f} >= "
                                  f"champion v{champ.version} {champ_val if champ_val is not None else 'n/a'}")
    else:
        promoted, reason = False, (f"val PR-AUC {new['val_pr_auc_final']:.4f} < "
                                   f"champion v{champ.version} {champ_val:.4f}")

    if promoted:
        client.set_registered_model_alias(name, alias, version)
    client.set_model_version_tag(name, version, "promotion", f"{'promoted' if promoted else 'rejected'}: {reason}")

    decision = {"name": name, "version": version, "alias": alias, "promoted": promoted, "reason": reason,
                "champion_version": version if promoted else (champ.version if champ else None),
                "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open("outputs/promotion.json", "w") as f:
        json.dump(decision, f, indent=4)
    print(f"{name} v{version}: {'PROMOTED to @' + alias if promoted else 'not promoted'} ({reason})")
