from pathlib import Path
import pandas as pd
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import torch
from chemprop import data, featurizers, models, nn

chemprop_dir = Path.cwd()
input_path = chemprop_dir / 'data' / 'delaney-processed.csv'
num_workers = 0
smiles_column = 'smiles'
target_columns = ['measured log solubility in mols per litre']

df_input = pd.read_csv(input_path)

smis = df_input.loc[:, smiles_column].values
ys = df_input.loc[:, target_columns].values

all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

mols = [d.mol for d in all_data]
train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1), seed=42)
train_data, val_data, test_data = data.split_data_by_indices(all_data, train_indices, val_indices, test_indices)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_dset = data.MoleculeDataset(train_data[0], featurizer)
scaler = train_dset.normalize_targets()

val_dset = data.MoleculeDataset(val_data[0], featurizer)
val_dset.normalize_targets(scaler)

test_dset = data.MoleculeDataset(test_data[0], featurizer)

train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

mp = nn.BondMessagePassing()
agg = nn.MeanAggregation()
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn = nn.RegressionFFN(output_transform=output_transform)
batch_norm = True
metrics_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

mpnn = models.MPNN(mp, agg, ffn, batch_norm=batch_norm, metrics=metrics_list)

# Configure Model Checkpoint
checkpointing = ModelCheckpoint(
    "checkpoints",
    "best-{epoch}-{val_loss:.2f}",
    "val_loss",
    mode="min",
    save_last=True
)

trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=30,
    callbacks=[checkpointing]
)

trainer.fit(mpnn, train_loader, val_loader)

results = trainer.test(dataloaders=test_loader, weights_only=False)
print(results)