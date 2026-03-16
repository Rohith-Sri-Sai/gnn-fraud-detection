from src.data_engine import DataEngine
from src.trainer import GraphTrainer

def main():
    # Phase 1: Data Engineering
    print("\n" + "="*50)
    print("PHASE 1: DATA ENGINEERING")
    print("="*50)
    
    # Initialize the engine with paths to your data
    engine = DataEngine(
        tx_path='data/train_transaction.csv', 
        id_path='data/train_identity.csv'
    )
    
    # Run the ETL pipeline and get the PyTorch Geometric HeteroData object
    data = engine.run()
    
    # Phase 2: Model Training & Evaluation
    print("\n" + "="*50)
    print("PHASE 2: GNN TRAINING")
    print("="*50)
    
    # Initialize the Trainer
    # hidden_channels=64 is a good starting point for learning capacity
    trainer = GraphTrainer(
        data=data, 
        hidden_channel=128, 
        batch_size=1024, 
        epochs=20
    )
    
    # Run the training loop
    trainer.run()

if __name__ == "__main__":
    main()