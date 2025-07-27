import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from model import OptionPricingModel
from dataset import OptionDataset, data_process
import matplotlib.pyplot as plt
import os

def load_model(model_path, device):
    """Load the trained model"""
    model = OptionPricingModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def evaluate_model(model, test_dataloader, device, target_min_max, target_pt):
    """Evaluate the model on test data"""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for log_ret, features, target in test_dataloader:
            log_ret = log_ret.to(device)
            features = features.to(device)
            target = target.view(-1, 1)
            
            output = model(log_ret, features)
            
            # Inverse transform predictions and targets to original scale
            pred_original_scale = target_min_max.inverse_transform(
                pd.DataFrame(output.cpu().numpy(), columns=['V'])
            )
            target_original_scale = target_min_max.inverse_transform(
                pd.DataFrame(target.cpu().numpy(), columns=['V'])
            )
            pred_original = target_pt.inverse_transform(
                pd.DataFrame(pred_original_scale, columns=['V'])
            )
            target_original = target_pt.inverse_transform(
                pd.DataFrame(target_original_scale, columns=['V'])
            )

            predictions.extend(pred_original.flatten())
            targets.extend(target_original.flatten())
    
    return np.array(predictions), np.array(targets)

def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    
    # Calculate percentage errors
    mape = np.mean(np.abs((targets - predictions) / targets)) * 100
    
    return {
        'MAPE': mape,
    }

def plot_buckets(preds, actuals, moneyness, maturity):
    df = pd.DataFrame({
        "Predicted": preds,
        "Actual": actuals,
        "Moneyness": moneyness,
        "Maturity": maturity
    })
    df["Percent_Error"] = 100 * ((df["Predicted"] - df["Actual"]) / df["Actual"])
    
    if len(df) > 2000:
        df = df.sample(2000, random_state=42)

    plt.figure(figsize=(12, 6))
    plt.scatter(df["Moneyness"], df["Percent_Error"], alpha=0.5, s=10)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel("Moneyness (S/K)")
    plt.ylabel("Percent Error (%)")
    plt.title("Percent Error vs. Moneyness")
    plt.savefig(os.path.join("inference_plots", "perror_moneyness.png"))
    plt.tight_layout()
    plt.show()

    # Plot percent error vs. maturity
    plt.figure(figsize=(12, 6))
    plt.scatter(df["Maturity"], df["Percent_Error"], alpha=0.5, s=10)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel("Maturity (T)")
    plt.ylabel("Percent Error (%)")
    plt.title("Percent Error vs. Maturity")
    plt.savefig(os.path.join("inference_plots", "perror_maturity.png"))
    plt.tight_layout()
    plt.show()
   


def plot_error_vs_price(preds, actuals):
    df = pd.DataFrame({
        "Predicted": preds,
        "Actual": actuals
    })
        
    if len(df) > 2000:
        df = df.sample(2000, random_state=42)

    df["Percent_error"] = 100 * (df["Predicted"] - df["Actual"]) / df["Actual"]

    plt.figure(figsize=(12, 6))
    plt.scatter(df["Actual"], df["Percent_error"], alpha=0.5, s=10)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel("Actual Option Price (V)")
    plt.ylabel("Percent Error (%)")
    plt.title("Percent Error vs. Actual Price")
    plt.savefig(os.path.join("inference_plots", "price_error.png"))
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    _, _, test_df, returns_df, target_min_max, target_pt, _ = data_process(bins=20, alpha=0.2, beta=0.7)
    # Create test dataset 
    moneyness = np.array(test_df["S"] / test_df["K"])
    maturities = np.array(test_df["T"])

    test_dataset = OptionDataset(test_df, returns_df, seq_length=140)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("option_pricing_model_final.pt", device)
    
    print(f"Testing on {len(test_dataset)} samples")
    
    # Evaluate model
    predictions, targets = evaluate_model(model, test_dataloader, device, target_min_max, target_pt)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Save results
    results_df = pd.DataFrame({
        'Actual': targets,
        'Predicted': predictions,
        'Error': targets - predictions
    })
    
    results_df.to_csv("inference_results.csv", index=False)
    print(f"\nResults saved to inference_results.csv")

    #visualize where the errors are worst
    plot_buckets(predictions, targets, moneyness, maturities)
    plot_error_vs_price(predictions, targets)

if __name__ == "__main__":
    main()
