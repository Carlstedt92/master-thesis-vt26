"""Import data and create batches for Chemprop model training."""

import pandas as pd
from chemprop import data, featurizers
from graph_augmentation import DINOMoleculeDataset, GraphAugmentation


class ChempropDataHandler:
    def __init__(self, input_path:str, num_workers:int=0, smiles_column:str="smiles"):
        self.input_path = input_path
        self.num_workers = num_workers
        self.smiles_column = smiles_column

    def read_data(self):
        df_input = pd.read_csv(self.input_path)
        #tdc_smiles = pd.read_csv("tdc_smiles.csv") # Smiles to exclude from dataset to avoid data leakage
        smis = df_input.loc[:, self.smiles_column].values
        #smis = [smi for smi in smis if smi not in tdc_smiles['Smiles'].values] # Exclude TDC smiles. Note: tdc_smiles.csv must be in the working directory and have a column named 'Smiles'
        all_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]
        self.all_data = [d for d in all_data if d.mol is not None]  # Filter out invalid molecules
        
    
    def create_ssl_dataloader(self, batch_size:int=32):
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer() # Maybe change for different featurization?
        base_dataset = data.MoleculeDataset(self.all_data, featurizer=None)
        dataset = DINOMoleculeDataset(base_dataset, transform= GraphAugmentation(local_views=4))
        dataset.featurizer = featurizer  # Set featurizer for the dataset
        self.dataloader = data.build_dataloader(dataset, batch_size=batch_size, num_workers=self.num_workers, shuffle=False)
        return self.dataloader
    

if __name__ == "__main__":
    input_path = 'data/delaney-processed.csv'
    data_handler = ChempropDataHandler(input_path)
    data_handler.read_data()
    print("Number of molecules loaded:", len(data_handler.all_data))
    dataloader = data_handler.create_ssl_dataloader(batch_size=64)
    print(f"Number of batches created: {len(dataloader)}")
    print("First batch:", next(iter(dataloader))[0])
