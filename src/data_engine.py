import pandas as pd
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataEngine:
    def __init__(self,tx_path,id_path):
        self.tx_path=tx_path
        self.id_path=id_path

        self.df=None
        self.data= HeteroData()
        self.scaler=StandardScaler()

        # Track our column names
        self.num_cols = ["TransactionAmt", "dist1", "dist2", "log_time_delta"] + [f"C{i}" for i in range(1, 15)]
        self.cat_cols = ["ProductCD", "card4", "card6"]
        self.final_features = []

    def run(self)-> HeteroData:
        print("1. Loading and Merging Data...")
        self._load_and_merge()

        print("\n2. Engineering Features...")
        self._engineer_features()

        print("\n3. Preprocesing...")
        self._preprocessing()

        print("\n4. Building Graph Topology...")
        self._build_graph_topology()

        print("\n5. Splitting Data (Strict Temporal)...")
        self._temporal_split()

        print("\n6. Scaling Features (Train-Only)...")
        self._scale_features()

        print("\n7. Finalizing Feature Tensor...")
        self._build_transaction_features()

        print(f"\n Graph Ready! Nodes: {self.data.num_nodes}, Edges: {self.data.num_edges}")
        return self.data

    def _load_and_merge(self):
        tx_df=pd.read_csv(self.tx_path)
        id_df=pd.read_csv(self.id_path)
        self.df=tx_df.merge(id_df, how='left',on='TransactionID')
    
    def _engineer_features(self):
        # sort by card1 and time, find gap between the transaction 
        self.df=self.df.sort_values(["card1", "TransactionDT"])

        # diff() easily calculates the difference between current and previous row
        self.df["time_delta"] = self.df.groupby("card1")["TransactionDT"].diff()

        # Fill the first transaction's NaN with 0, and ensure no negative times
        self.df["time_delta"] = self.df["time_delta"].fillna(0).clip(lower=0)
        self.df["log_time_delta"] = np.log1p(self.df["time_delta"])

    def _preprocessing(self):
        self.num_cols=[c for c in self.num_cols if c in self.df.columns]
        self.cat_cols=[c for c in self.cat_cols if c in self.df.columns]
        self.final_features=self.num_cols+self.cat_cols

        self.df[self.num_cols]=self.df[self.num_cols].fillna(self.df[self.num_cols].median())

        encoder=LabelEncoder()
        for cols in self.cat_cols:
            self.df[cols]=encoder.fit_transform(self.df[cols].astype(str))

    def _build_graph_topology(self):
        # Re-sort chronologically for the Train/Test split later
        self.df=self.df.sort_values("TransactionDT").reset_index(drop=True)

        # Setup Transaction Nodes
        self.data["transaction"].y=torch.tensor(self.df["isFraud"].values, dtype=torch.long)
        self.data["transaction"].time=torch.tensor(self.df["TransactionDT"].values, dtype=torch.long)
        self.data["transaction"].num_nodes = len(self.df)

        # Setup Card Nodes & Edges (pd.factorize converts raw IDs to 0,1,2... cleanly)
        self.df["card_idx"], _ = pd.factorize(self.df["card1"])
        self.data["card"].num_nodes = self.df["card_idx"].nunique()

        src = torch.arange(len(self.df))
        dst = torch.tensor(self.df["card_idx"].values, dtype=torch.long)
        self.data["transaction", "uses", "card"].edge_index = torch.stack([src, dst], dim=0)

        # Allow messages to flow both ways
        self.data = T.ToUndirected()(self.data)
    
    def _temporal_split(self):
        # 70% Train, 15% Validation, 15% Test
        n =len(self.df)
        train_end=int(0.7*n)
        val_end=int(0.85*n)

        train_mask=torch.zeros(n,dtype=torch.bool)
        val_mask= torch.zeros(n,dtype=torch.bool)
        test_mask=torch.zeros(n,dtype=torch.bool)

        train_mask[:train_end]=True
        val_mask[train_end:val_end]=True
        test_mask[val_end:]=True

        self.data["transaction"].train_mask= train_mask
        self.data["transaction"].val_mask=val_mask
        self.data["transaction"].test_mask=test_mask

    def _scale_features(self):
        # Get indices for training data
        train_idx=self.data["transaction"].train_mask.numpy()

        # Use scaler fit for ONLY train data
        self.scaler.fit(self.df.loc[train_idx, self.num_cols])

        # Apply the scale to everything
        self.df[self.num_cols]=self.scaler.transform(self.df[self.num_cols])

    def _build_transaction_features(self):
        # Isolate the training data
        train_mask=self.data["transaction"].train_mask.numpy()
        train_df=self.df.loc[train_mask]

        card_stats=train_df.groupby("card1").agg({
            "TransactionAmt":["mean","std"],
            "TransactionID":"count"
        }).fillna(0)
        # Flatten multi index columns
        card_stats.columns=["amt_mean","amt_std","tx_count"]

        # Scale these features for neural network stability
        # log1p safely handles 0s and massive outliers
        card_stats["amt_mean"]=np.log1p(card_stats["amt_mean"])
        card_stats["amt_std"]=np.log1p(card_stats["amt_std"])
        card_stats["tx_count"]=np.log1p(card_stats["tx_count"])

        # Map those card_stats with our pytorch indices
        mapping_df=self.df[["card1","card_idx"]].drop_duplicates().set_index("card1")
        aligned_stats=mapping_df.join(card_stats,how="left").sort_values("card_idx")

        feature_cols=["amt_mean","amt_std","tx_count"]
        aligned_stats[feature_cols]=aligned_stats[feature_cols].fillna(0)

        # Assign Features to the 'Card' Nodes
        self.data["card"].x = torch.tensor(
            aligned_stats[feature_cols].values, 
            dtype=torch.float32
        )
        # Assign Features to the 'Transaction' Nodes
        self.data["transaction"].x=torch.tensor(
            self.df[self.final_features].values,
            dtype=torch.float32
            )


if __name__ == "__main__":
    engine=DataEngine("data/train_transaction.csv","data/train_identity.csv")
    g=engine.run()


