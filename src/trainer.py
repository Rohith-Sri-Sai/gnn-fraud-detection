import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from src.model import GraphModel
import os
import json
from src.utils import load_params
import mlflow
import mlflow.pytorch
import subprocess

class GraphTrainer:
    def __init__(self,data, hidden_channel=128,batch_size=1024, epochs=20):
        self.data=data
        self.epochs=epochs
        self.batch_size=batch_size

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {self.device}")

        # 1. Initialize Model & Optimizer
        self.model= GraphModel(
            hidden_channels=hidden_channel,
            out_channels=2,
            metadata=data.metadata()
        ).to(self.device)

        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=0.005)

        # 2. Handle the 97% / 3% Class Imbalance
        # We can add weights to loss
        # Misclassifying a fraud(class1), the penalty is 10 times more than misclassifying a legit(class0)
        weights=torch.tensor([1.0,10.0],dtype=torch.float32).to(self.device)
        self.loss_fn=torch.nn.CrossEntropyLoss(weight=weights)

        # 3. Create the Mini-Batch Loaders
        self.train_loader = self._create_loader('train_mask', shuffle=True)
        self.val_loader = self._create_loader('val_mask', shuffle=False)
        self.test_loader = self._create_loader('test_mask',shuffle=False)

    def _create_loader(self, mask_name, shuffle):
        """
        Creates a NeighborLoader to sample sub-graphs.
        Instead of loading the whole graph, we pick a batch of transactions,
        grab 15 of their immediate neighbors, and 10 of their neighbors' neighbors.
        """
        return NeighborLoader(
            self.data,
            # Sample 15 neighbors for Layer 1, and 10 neighbors for Layer 2
            num_neighbors=[15, 10], 
            batch_size=self.batch_size,
            # Only start sampling from transactions that belong to this specific mask
            input_nodes=('transaction', self.data['transaction'][mask_name]),
            shuffle=shuffle
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            # Move the mini-batch to the GPU
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward Pass: Get predictions for the batch
            out = self.model(batch.x_dict, batch.edge_index_dict)
            
            # CRITICAL FAANG CONCEPT: Slicing the Batch
            # The loader pulls in our target transactions PLUS all their neighbors.
            # We only want to calculate the loss on the actual target transactions!
            batch_size = batch['transaction'].batch_size
            
            pred = out[:batch_size]
            target = batch['transaction'].y[:batch_size]
            
            # Calculate Loss and Update Weights
            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    @torch.no_grad() # Turn off gradients to save memory during evaluation
    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x_dict, batch.edge_index_dict)
            
            batch_size = batch['transaction'].batch_size
            
            # Get the probability that the transaction is Fraud (Class 1)
            # Softmax converts raw output numbers into percentages (0.0 to 1.0)
            probabilities = F.softmax(out[:batch_size], dim=1)[:, 1]
            targets = batch['transaction'].y[:batch_size]
            
            all_preds.append(probabilities.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
        # Combine all batches into one giant array
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Calculate Metrics
        roc_auc = roc_auc_score(all_targets, all_preds)
        pr_auc = average_precision_score(all_targets, all_preds) # PR-AUC is better for imbalance
        
        return roc_auc, pr_auc

    def run(self):
        print("\n Starting Training Loop...")

        for epoch in range(1, self.epochs + 1):
            loss = self.train_epoch()
            val_roc, val_pr = self.evaluate(self.val_loader)

            mlflow.log_metric(
                "train_loss",
                loss,
                step=epoch
            )

            mlflow.log_metric(
                "val_roc_auc",
                val_roc,
                step=epoch
            )

            mlflow.log_metric(
                "val_pr_auc",
                val_pr,
                step=epoch
            )

            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {loss:.4f} | "
                f"Val ROC-AUC: {val_roc:.4f} | "
                f"Val PR-AUC: {val_pr:.4f}"
            )

        print("\n" + "=" * 50)
        print(" FINAL PRODUCTION TEST EVALUATION ")
        print("=" * 50)

        test_roc, test_pr = self.evaluate(self.test_loader)

        print(f"Final Test ROC-AUC: {test_roc:.4f}")
        print(f"Final Test PR-AUC:  {test_pr:.4f}")

        return test_roc, test_pr

if __name__ == "__main__":
    params = load_params()["train"]
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Fraud Detection GNN")
    
    # 1. Load the processed graph
    data = torch.load("data/processed/graph.pt")
    
    # 2. Initialize and run trainer
    with mlflow.start_run():

        trainer = GraphTrainer(
            data=data,
            hidden_channel=params["hidden_channel"],
            batch_size=params["batch_size"],
            epochs=params["epochs"]
        )
        mlflow.log_params(params)

        test_roc, test_pr = trainer.run()

        # 3. Save model weights
        os.makedirs("models", exist_ok=True)
        torch.save(trainer.model.state_dict(), "models/graph_sage_model.pt")
        mlflow.pytorch.log_model(
            trainer.model,
            artifact_path="model"
        )
        
        # 4. Evaluate and save metrics
        mlflow.log_metric("test_roc_auc", test_roc)
        mlflow.log_metric("test_pr_auc", test_pr)
        metrics = {
            "test_roc_auc": float(test_roc),
            "test_pr_auc": float(test_pr)
        }
        
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact("params.yaml")
    print("Saved model weights and metrics.json")
