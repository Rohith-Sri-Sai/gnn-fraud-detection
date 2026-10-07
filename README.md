# GNN Fraud Detection on T-Finance

A GraphSAGE model that flags anomalous accounts (fraud, money laundering, gambling) in
**T-Finance**, a real financial transaction graph. Training runs full-batch on a GPU, the
pipeline is reproducible with **DVC**, and every run is tracked in **MLflow** and registered
in the MLflow Model Registry (PostgreSQL backend, persistent artifact store, Docker Compose).
A promotion stage moves the `@champion` alias to a new version only if it's at least as good,
and a **FastAPI** service in Docker serves fraud scores and alerts from the champion model.

| Metric (test set, 5-model ensemble) | Result |
|---|---|
| **PR-AUC** | **0.906 ± 0.002** (mean ± std over 5 random splits) |
| ROC-AUC | 0.975 |
| PR-AUC without the graph (MLP / XGBoost baselines) | 0.814 |

---

## Dataset

**T-Finance** (Tang et al., *Rethinking Graph Neural Networks for Anomaly Detection*, ICML 2022),
also part of the GADBench benchmark.

| Property | Value |
|---|---|
| Nodes (accounts) | 39,357 |
| Edges | 21.2M undirected (42.4M directed, stored both ways) |
| Features per account | 10 (heavy-tailed counts and amounts) |
| Anomalous accounts | 1,804 (4.6%) |
| Average / median degree | 1,078 / 265 |
| Timestamps | none, so the benchmark uses a stratified random split |

The graph carries a strong signal: 47% of an anomalous account's neighbors are anomalous,
against 1.5% for a normal account.

---

## Project structure

```
├── src/
│   ├── data_engine.py       # DVC stage 1: split, scale features, build the graph
│   ├── trainer.py           # DVC stage 2: full-batch training, alert threshold, MLflow registry
│   ├── promote.py           # DVC stage 3: move the @champion alias if the new model is as good
│   ├── model.py             # GraphSAGE model + seed-ensemble wrapper
│   ├── features.py          # label-trick input features (shared by training and scoring)
│   ├── predict.py           # batch scoring with the @champion model
│   ├── api.py               # FastAPI service (scores, alerts, reload)
│   ├── convert_tfinance.py  # one-time: DGL file -> plain tensors (needs dgl)
│   └── utils.py             # params.yaml loader
├── params.yaml              # all preprocessing, training, alert and promotion settings
├── dvc.yaml / dvc.lock      # pipeline definition + locked hashes
├── docker-compose.yml       # PostgreSQL + MLflow server + scoring API
├── docker/Dockerfile.mlflow
├── docker/Dockerfile.api
├── requirements.txt
└── .env.example             # database credentials template
```

---

## Setup

### 1. Python environment (Python 3.11, CUDA 12.8, pip)

PyTorch is built for CUDA 12.8, which Blackwell GPUs (sm_120) require. It also works on older NVIDIA GPUs.

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

Check the GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Start MLflow (tracking server + model registry)

```bash
cp .env.example .env      # then set a real POSTGRES_PASSWORD in .env (gitignored)
docker compose up -d --build
curl http://localhost:5000/health   # -> OK
```

This starts PostgreSQL, the MLflow server and the scoring API. The API only becomes healthy
once a `@champion` model exists, so on a fresh setup run the pipeline first (next sections),
then `docker compose up -d api`.

- **UI:** http://localhost:5000. Registered models are under the **Models** tab.
- **Localhost only:** MLflow (port 5000) and the API (port 8000) accept connections only from the server itself, because neither has authentication. From your laptop, open an SSH tunnel and then browse `http://localhost:5000` and `http://localhost:8000/docs`:
  ```bash
  ssh -L 5000:localhost:5000 -L 8000:localhost:8000 <user>@<server>
  ```
- **Persistence:** run metadata lives in the `postgres_data` Docker volume and artifacts in `./artifacts/`. Both survive `docker compose down` and restarts. Don't use `docker compose down -v`, which deletes the database volume.

### 3. Get the data

Download `tfinance.zip` from the authors' Google Drive, linked from
<https://github.com/squareRoot3/Rethinking-Anomaly-Detection>, and unzip it into `data/raw/tfinance/`.

The file is a DGL graph. Convert it once to plain tensors, so the project itself doesn't depend on DGL.
Use a separate Python environment, because DGL doesn't support the project's torch version:

```bash
python3.11 -m venv dglenv
source dglenv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install "dgl==2.4.0" \
  -f https://data.dgl.ai/wheels/torch-2.4/repo.html "numpy<2" pandas pyyaml pydantic packaging requests tqdm psutil
python -m src.convert_tfinance data/raw/tfinance/tfinance data/raw/tfinance/tfinance.pt
deactivate
```

---

## Run the pipeline

```bash
source venv/bin/activate
dvc repro          # preprocess -> train -> promote
dvc metrics show   # test PR-AUC / ROC-AUC, alert precision/recall from metrics.json
```

Takes about 10 minutes on one GPU: 5 models × 500 full-batch epochs, about 100 s each.

Each run:
- creates an MLflow run in the experiment **`Fraud Detection GNN - tfinance`** with all parameters, graph size, per-epoch train loss and validation metrics, per-member and ensemble test metrics, and tags (GPU, CUDA, torch version, git commit);
- picks an **alert threshold** on the validation set (the lowest score with at least 90% precision) and reports its precision and recall on the test set;
- registers a **new version** of the model **`gnn-fraud-tfinance-graphsage`**, tagged with its threshold and scores; earlier versions are kept;
- **promotes** it (`@champion` alias) if its test PR-AUC is at least 0.90 and its validation PR-AUC is at least the current champion's; the decision is written to `outputs/promotion.json` and as a tag on the version;
- writes `models/graph_sage_model.pt` and `metrics.json`, both tracked by DVC.

Optional environment overrides:

| Variable | Default |
|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` |
| `MLFLOW_EXPERIMENT_NAME` | `Fraud Detection GNN - tfinance` |
| `MLFLOW_REGISTERED_MODEL_NAME` | `gnn-fraud-tfinance-graphsage` |

Change settings in `params.yaml`, then run `dvc repro` again. DVC re-runs only the stages
whose inputs changed.

---

## Method

1. **Split:** a stratified random split of 40% train, 20% validation and 40% test (the benchmark protocol), controlled by `split_seed`.
2. **Features:** `log1p`, then standardization using statistics from training accounts only.
3. **Model:** 2-layer GraphSAGE (mean aggregation) with an input projection, LayerNorm, ReLU, dropout 0.3 and residual connections, then a linear classifier. Hidden size 128.
4. **Full-batch training:** the whole graph sits on the GPU as a sparse adjacency matrix (about 2 GB). Each epoch is one AdamW step on all training accounts (lr 0.003, weight decay 1e-4, 500 epochs).
5. **Label trick** (Wang et al., 2021):
   - each account gets two extra input features, `[is_known, is_fraud]`, filled from **training labels only**;
   - every epoch, half of the training accounts expose their label and the loss is computed on the other half, so an account never sees its own label;
   - at evaluation, all training labels are exposed; validation and test labels are never inputs.
6. **Loss:** unweighted cross-entropy (`pos_weight_power: 0`). Class weighting raised recall but lowered precision, and so lowered PR-AUC.
7. **Model selection:** each model keeps the checkpoint with the best validation PR-AUC.
8. **Ensemble:** 5 seeds (42–46), with their fraud probabilities averaged. The ensemble is registered as a single `EnsembleModel`.

Tuning used **validation PR-AUC only**, averaged over split seeds 0–2 (34 configurations).
The test set was used only for final scoring.

---

## Results

### Final model on 5 splits (5-model ensemble each)

| Split | Test PR-AUC | Test ROC-AUC |
|---|---|---|
| 0 | 0.908 | 0.975 |
| 1 | 0.907 | 0.975 |
| 2 | 0.909 | 0.975 |
| 3 (never used in tuning) | 0.903 | 0.976 |
| 4 (never used in tuning) | 0.904 | 0.974 |
| **Mean** | **0.906 ± 0.002** | **0.975** |

Single models average 0.904; the ensemble makes the result stable.

### What each step contributed (mean test PR-AUC)

| Model | PR-AUC |
|---|---|
| MLP, features only | 0.814 |
| XGBoost, features only | 0.814 |
| GraphSAGE, untuned | 0.897 |
| GraphSAGE, hyperparameter grid (22 configs) | 0.891–0.902 (plateau) |
| + label trick, unweighted loss, dropout 0.3, 500 epochs | ~0.905 |
| **+ 5-seed ensemble** | **0.906** |

Also tested, with no gain: hidden size 256, 3 layers, sum or max aggregation.
300 epochs was about 0.002 worse than 500 and dropped one split below 0.90.

---

## Using the registered model

The model expects the account features plus the two label-trick columns (training labels only),
and a sparse adjacency:

```python
import torch, mlflow, mlflow.pytorch
import torch_geometric.transforms as T

mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.pytorch.load_model("models:/gnn-fraud-tfinance-graphsage/latest").eval()

data = torch.load("data/processed/graph.pt", weights_only=False)
g = T.ToSparseTensor()(data)
acc = g["account"]
known = acc.train_mask.float()                       # labels we are allowed to use
x = {"account": torch.cat([acc.x, torch.stack([known, (acc.y == 1).float() * known], 1)], 1)}

with torch.no_grad():
    fraud_prob = torch.softmax(model(x, g.adj_t_dict), dim=1)[:, 1]
```

---

## Limitations

- **Random split, not time-based:** T-Finance has no timestamps. Real fraud changes over time, so a time-ordered evaluation would likely score lower.
- **Transductive setting:** the model scores accounts already in the graph and uses the known labels of their neighbors. A brand-new account would need its edges added and the model run again.
- **Anonymized features:** the 10 features aren't described in the dataset, so the model's decisions can't be explained in business terms.
