import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_loss_data(file_path):
    """
    Load loss data from a JSON file.

    Parameters:
    file_path (str): The path to the JSON file containing loss data.

    Returns:
    pd.DataFrame: A DataFrame containing the loss data.
    """
    return pd.read_json(file_path)

def plot_train_val_loss_curves(loss_data, output_path, model_name="Model"):
    """
    Plot training, validation

    Parameters:
    loss_data (pd.DataFrame): A DataFrame containing 'epoch', 'train_loss', 'val_loss'.
    output_path (str): The path to save the generated plot.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    plt.plot(loss_data['epoch'], loss_data['train_loss'], label='Training Loss', marker='o')
    
    # Plot validation loss if available
    if 'val_loss' in loss_data.columns and loss_data['val_loss'].notnull().any():
        plt.plot(loss_data['epoch'], loss_data['val_loss'], label='Validation Loss', marker='o')
    
        plt.title(f'Training and Validation Loss Curves for {model_name}')
    else:
        plt.title(f'Training Loss Curve for {model_name}')    
    plt.title(f'Loss Curves for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Example usage
    model = "GINE_DINO"
    path = f"models/{model}/loss_history.json"
    loss_data = load_loss_data(path)
    plot_loss_curves(loss_data, f"models/{model}/loss_curves.png", model_name=model)