import matplotlib.pyplot as plt
import json
from pathlib import Path


def _set_plot_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

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


def _extract_series(records, x_key, y_key):
    xs = []
    ys = []
    for row in records:
        if isinstance(row, dict) and x_key in row and y_key in row and row[y_key] is not None:
            xs.append(row[x_key])
            ys.append(row[y_key])
    return xs, ys


def _extract_online_knn_validation_metric(online_eval_entry: dict, dataset: str, preferred_metric: str = "rmse"):
    """Extract the online kNN validation metric for a dataset from one eval record."""
    datasets = online_eval_entry.get("evaluation", {}).get("datasets", {})
    dataset_payload = datasets.get(dataset)
    if dataset_payload is None:
        return None, None

    knn = dataset_payload.get("embeddings", {}).get("knn", {})
    validation_metrics = knn.get("validation_metrics", {})

    if preferred_metric and validation_metrics.get(preferred_metric) is not None:
        return validation_metrics.get(preferred_metric), f"knn_val_{preferred_metric}"

    primary_value = dataset_payload.get("primary_metric_value")
    primary_name = dataset_payload.get("primary_metric_name")
    if primary_value is not None and primary_name is not None:
        return primary_value, str(primary_name)

    if not validation_metrics:
        return None, None

    # Fallback order if primary metric fields are unavailable in older histories.
    for metric_name in (preferred_metric, "rmse", "r2", "mae", "roc_auc", "f1", "balanced_accuracy"):
        if not metric_name:
            continue
        metric_value = validation_metrics.get(metric_name)
        if metric_value is not None:
            return metric_value, f"knn_val_{metric_name}"

    return None, None


def plot_train_val_loss_curves(loss_data, output_path, model_name="Model"):
    """
    Plot training and validation loss curves (legacy format).

    Parameters:
    loss_data (pd.DataFrame or dict): Loss data containing 'epoch', 'train_loss', 'val_loss'.
    output_path (str): The path to save the generated plot.
    model_name (str): Name of the model for the title.
    """
    if isinstance(loss_data, dict):
        if "DINO_Loss" in loss_data:
            loss_data = loss_data["DINO_Loss"]
        else:
            loss_data = []
    
    _set_plot_style()
    plt.figure(figsize=(10, 6))
    
    epoch_values, train_losses = _extract_series(loss_data, 'epoch', 'train_loss')
    if epoch_values and train_losses:
        plt.plot(epoch_values, train_losses, label='Training Loss')
        
        # Plot validation loss if available
        val_epochs, val_losses = _extract_series(loss_data, 'epoch', 'val_loss')
        if val_epochs and val_losses:
            plt.plot(val_epochs, val_losses, label='Validation Loss')
            plt.title(f'Training and Validation Loss Curves for {model_name}')
        else:
            plt.title(f'Training Loss Curve for {model_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()


def plot_ssl_and_online_knn(loss_history_path, output_path, model_name="Model", dataset="lipo", preferred_metric="rmse"):
    """
    Plot SSL training loss and online kNN validation metric on dual axes.

    Parameters:
    loss_history_path (str or Path): Path to loss_history.json
    output_path (str or Path): Path where to save the plot
    model_name (str): Model name for title
    dataset (str): Dataset key in online eval (e.g., 'lipo' or 'hiv')
    """
    with open(loss_history_path, 'r') as f:
        loss_history = json.load(f)

    dino_loss = loss_history.get("DINO_Loss", [])
    online_eval = loss_history.get("Evaluation_Loss", {}).get("online_eval", [])

    if not dino_loss:
        raise ValueError("No DINO_Loss entries found in loss_history.json")

    dino_epochs, dino_train_losses = _extract_series(dino_loss, "epoch", "train_loss")
    if not dino_epochs or not dino_train_losses:
        raise ValueError("No train_loss entries found in DINO_Loss")

    # Extract online kNN metrics
    eval_rows = []
    metric_label = "online_val_metric"
    for entry in online_eval:
        value, detected_label = _extract_online_knn_validation_metric(entry, dataset, preferred_metric=preferred_metric)
        if value is None:
            continue
        if detected_label is not None:
            metric_label = detected_label
        eval_rows.append({"epoch": int(entry["epoch"]), "online_metric": float(value)})

    _set_plot_style()
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Plot SSL loss on left axis
    ax1.plot(dino_epochs, dino_train_losses, color="#1f77b4", label="ssl_train_loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SSL Train Loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    lines = []
    labels = []
    for line in ax1.get_lines():
        lines.append(line)
        labels.append(line.get_label())

    # Plot online evaluation metric on right axis if available
    if eval_rows:
        ax2 = ax1.twinx()
        eval_epochs = [row["epoch"] for row in eval_rows]
        eval_values = [row["online_metric"] for row in eval_rows]
        ax2.plot(eval_epochs, eval_values, color="#d62728", label=metric_label, linewidth=2)
        ax2.set_ylabel(metric_label, color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        for line in ax2.get_lines():
            lines.append(line)
            labels.append(line.get_label())

    plt.title(f"SSL and Online kNN Curves ({model_name}) - {dataset}")
    fig.legend(lines, labels, loc="upper right", frameon=True)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
if __name__ == "__main__":
    # Example usage
    model = "GDZ_GAT_TEST"
    path = f"models/{model}/loss_history.json"
    loss_data = load_loss_data(path)
    
    # Try to plot with online eval data first, fall back to simple loss curves
    try:
        plot_ssl_and_online_knn(
            path,
            f"models/{model}/loss_curves_ssl_knn.png",
            model_name=model,
            dataset="lipo",
            preferred_metric="rmse",
        )
    except (ValueError, KeyError):
        # Fall back to simple training/validation loss plot
        plot_train_val_loss_curves(loss_data, f"models/{model}/loss_curves.png", model_name=model)