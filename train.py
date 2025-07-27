import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from model import OptionPricingModel
from dataset import OptionDataset, data_process
import matplotlib.pyplot as plt

# ---- 2. Load Data ----

train_df, val_df, test_df, returns_df, _, _, = data_process(bins=20, alpha=0.2, beta=0.7)

train_dataset = OptionDataset(train_df, returns_df, seq_length=140)
val_dataset = OptionDataset(val_df, returns_df, seq_length=140)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print("---------Data completed---------")
# He initialization function
def he_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

#training specifications
#using a weighted loss to put emphasis on higher value options

learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OptionPricingModel().to(device)
print("-------Model instance created--------")
model.apply(he_init_weights)
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

# Early stopping parameters
early_stopping_patience = 3
epochs = 25
best_val_loss = float('inf')
best_model_state = None
epochs_no_improve = 0
train_losses = []
val_losses = []
batch_grads = []
print("-------Training begins--------")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for log_ret, features, target in train_dataloader:
        #training loop
        log_ret = log_ret.to(device)
        features = features.to(device)
        target = target.view(-1, 1).to(device)
        optimizer.zero_grad()
        output = model(log_ret, features)
        loss = criterion(output, target)
        loss.backward()

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        batch_grads.append(total_norm)

        optimizer.step()
        running_loss += loss.item() * target.size(0)
    avg_train_loss = running_loss / len(train_dataset)
    train_losses.append(avg_train_loss)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for log_ret, features, target in val_dataloader:
            #validation loop
            log_ret = log_ret.to(device)
            features = features.to(device)
            target = target.view(-1, 1).to(device)
            output = model(log_ret, features)
            loss = criterion(output, target)
            val_loss += loss.item() * target.size(0)
    avg_val_loss = val_loss / len(val_dataset)
    val_losses.append(avg_val_loss)
    # Step the scheduler
    scheduler.step(avg_val_loss)
    for param_group in optimizer.param_groups:
       print(f"Current learning rate: {param_group['lr']}")
    # early stopping after 3 bad epochs
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

# Save the best model
torch.save(best_model_state, "option_pricing_model_final.pt")
print(f"Best model saved to option_pricing_model.pt (Best Val Loss: {best_val_loss:.6f})")

epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.title("Learning Curve")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig("training_images/learning_curve.png", bbox_inches='tight')

plt.show()

plt.figure(figsize=(10, 4))
plt.plot(batch_grads, label='Gradient Norm')
plt.xlabel('Batch')
plt.ylabel('L2 Norm of Gradients')
plt.title('Gradient Norms During Training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_images/gradients.png", bbox_inches='tight')
plt.show()