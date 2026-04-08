import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path

def load_loss_data(file_path):
    """
    Load loss data from a JSON file.

    Parameters:
    file_path (str): The path to the JSON file containing loss data.

    Returns:
    dict or pd.DataFrame: The loaded loss data structure.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def _extract_linear_probe_validation_metric(online_eval_entry: dict, dataset: str):
    """Extract linear probe validation metric from online eval entry."""
    datasets = online_eval_entry.get("evaluation", {}).get("datasets", {})
    dataset_payload = datasets.get(dataset)
    if dataset_payload is None:
        return None, None

    linear_probe = dataset_payload.get("embeddings", {}).get("linear_probe", {})
    validation_metrics = linear_probe.get("validation_metrics", {})
    if not validation_metrics:
        return None, None

    # Regression uses RMSE, classification uses accuracy
    rmse = validation_metrics.get("rmse")
    if rmse is not None:
        return rmse, "linear_probe_val_rmse"
    
    accuracy = validation_metrics.get("accuracy")
    if accuracy is not None:
        return accuracy, "linear_probe_val_accuracy"
    
    return None, None


def plot_train_val_loss_curves(loss_data, output_path, model_name="Model"):
    """
    Plot training and validation loss curves (legacy format).

    Parameters:
    loss_data (pd.DataFrame or dict): Loss data containing 'epoch', 'train_loss', 'val_loss'.
    output_path (str): The path to save the generated plot.
    model_name (str): Name of the model for the title.
    """
    # Handle dict format (old-style loss history)
    if isinstance(loss_data, dict):
        if "DINO_Loss" in loss_data:
            loss_data = pd.DataFrame(loss_data["DINO_Loss"])
        else:
            loss_data = pd.DataFrame(loss_data)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    if 'epoch' in loss_data.columns and 'train_loss' in loss_data.columns:
        plt.plot(loss_data['epoch'], loss_data['train_loss'], label='Training Loss', marker='o')
        
        # Plot validation loss if available
        if 'val_loss' in loss_data.columns and loss_data['val_loss'].notnull().any():
            plt.plot(loss_data['epoch'], loss_data['val_loss'], label='Validation Loss', marker='o')
            plt.title(f'Training and Validation Loss Curves for {model_name}')
        else:
            plt.title(f'Training Loss Curve for {model_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()


def plot_ssl_and_linear_probe(loss_history_path, output_path, model_name="Model", dataset="lipo"):
    """
    Plot SSL training loss and linear probe validation metric on dual axes.

    Parameters:
    loss_history_path (str or Path): Path to loss_history.json
    output_path (str or Path): Path where to save the plot
    model_name (str): Model name for title
    dataset (str): Dataset key in online eval (e.g., 'lipo' or 'hiv')
    """
    with open(loss_history_path, 'r') as f:
        loss_history = json.load(f)

    dino_loss = pd.DataFrame(loss_history.get("DINO_Loss", []))
    online_eval = loss_history.get("Evaluation_Loss", {}).get("online_eval", [])

    if dino_loss.empty:
        raise ValueError("No DINO_Loss entries found in loss_history.json")

    # Extract linear probe metrics
    linear_rows = []
    metric_label = "linear_probe_val_metric"
    for entry in online_eval:
        value, detected_label = _extract_linear_probe_validation_metric(entry, dataset)
        if value is None:
            continue
        if detected_label is not None:
            metric_label = detected_label
        linear_rows.append({"epoch": int(entry["epoch"]), "linear_probe_metric": float(value)})

    linear_df = pd.DataFrame(linear_rows)

    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Plot SSL loss on left axis
    ax1.plot(
        dino_loss["epoch"],
        dino_loss["train_loss"],
        color="#1f77b4",
        label="ssl_train_loss",
        linewidth=2,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SSL Train Loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    lines = []
    labels = []
    for line in ax1.get_lines():
        lines.append(line)
        labels.append(line.get_label())

    # Plot linear probe metric on right axis if available
    if not linear_df.empty:
        ax2 = ax1.twinx()
        ax2.plot(
            linear_df["epoch"],
            linear_df["linear_probe_metric"],
            color="#d62728",
            label=metric_label,
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax2.set_ylabel(metric_label, color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        for line in ax2.get_lines():
            lines.append(line)
            labels.append(line.get_label())

    plt.title(f"SSL and Linear Probe Curves ({model_name}) - {dataset}")
    fig.legend(lines, labels, loc="upper right", frameon=True)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    # Example usage
    model = "GINE_DINO"
    path = f"models/{model}/loss_history.json"
    loss_data = load_loss_data(path)
    
    # Try to plot with online eval data first, fall back to simple loss curves
    try:
        plot_ssl_and_linear_probe(path, f"models/{model}/loss_curves_ssl_linearprobe.png", model_name=model, dataset="lipo")
    except (ValueError, KeyError):
        # Fall back to simple training/validation loss plot
        if isinstance(loss_data, dict) and "DINO_Loss" in loss_data:
            loss_data = pd.DataFrame(loss_data["DINO_Loss"])
        plot_train_val_loss_curves(loss_data, f"models/{model}/loss_curves.png", model_name=model)