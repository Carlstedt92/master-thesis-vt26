import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import seaborn

# Fetch loss history from training logs
with open("models/GINE_DINO_ZINC/loss_history.json", "r") as f:
    loss_history = json.load(f)

# Convert to DataFrame for plotting
dino_loss_df = pd.DataFrame(loss_history["DINO_Loss"])

# Plotting
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
plt.plot(dino_loss_df["epoch"], dino_loss_df["train_loss"], label="DINO Training Loss", marker="o")
plt.title("DINO Loss Curves for GINE_DINO_ZINC")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("models/GINE_DINO_ZINC/loss_curves_log_y.png")