# %%
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from huggingface_hub import hf_hub_download
import scipy.linalg as linalg

from .utils import get_sae, LinearAdapter

# %%

BIG_MODEL = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sae = get_sae(big_model=BIG_MODEL)

hp = "blocks.12.hook_resid_post"
def download_effects_data():
    """Download pre-computed effects data from HuggingFace."""
    repo_id = "schalnev/sae-ts-effects"
    filename = "effects_9b.pt" if BIG_MODEL else "effects_2b.pt"
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        data = torch.load(path)
        return data['features'], data['effects']
    except Exception as e:
        print(f"Error downloading effects data: {e}")
        raise

def get_training_data():
    """Get training data either from local files or download from HuggingFace."""
    local_path = "effects.pt"
    
    if os.path.exists(local_path):
        print("Using local effects file")
        data = torch.load(local_path)
        features, effects = data['features'], data['effects']
    else:
        print("Downloading effects file from HuggingFace")
        features, effects = download_effects_data()
    
    # Normalize features to have norm 1
    features = features / torch.norm(features, dim=-1, keepdim=True)
    
    return features, effects

def calculate_rotation_matrix(adapter: LinearAdapter, sae):
    """Calculate rotation matrix between adapter and SAE decoder weights."""
    # Normalize weights
    normed_adapter = adapter.W / torch.norm(adapter.W, dim=0)
    normed_decoder = sae.W_dec / torch.norm(sae.W_dec, dim=1, keepdim=True)
    
    # Convert to numpy for SVD calculation
    adapter_np = normed_adapter.detach().cpu().numpy()
    decoder_np = normed_decoder.detach().cpu().numpy()
    
    # Calculate rotation matrix using SVD
    # M = decoder_np @ adapter_np.T
    M = adapter_np @ decoder_np
    U, _, Vt = linalg.svd(M)
    R = U @ Vt
    
    return torch.tensor(R).to(device)

def calculate_correction_bias(adapter: LinearAdapter):
    """Calculate correction bias from adapter."""
    b = adapter.W @ adapter.b
    b = b / torch.norm(b)
    return b

def train(num_epochs, lr=1e-4):
    features, effects = get_training_data()

    n_val = 100

    val_features = features[-n_val:]
    val_effects = effects[-n_val:]
    features = features[:-n_val]
    effects = effects[:-n_val]

    dataset = TensorDataset(features, effects)
    val_dataset = TensorDataset(val_features, val_effects)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    adapter = LinearAdapter(sae.W_enc.shape[0], sae.W_enc.shape[1]).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(opt, T_max=num_epochs)

    for epoch in range(num_epochs):
        adapter.train()
        total_loss = 0
        num_batches = 0

        for batch_features, batch_effects in dataloader:
            opt.zero_grad()
            batch_features = batch_features.to(device)
            batch_effects = batch_effects.to(device)
            pred = adapter(batch_features)
            loss = F.mse_loss(pred, batch_effects)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        # Validation
        adapter.eval()
        val_total_loss = 0
        val_num_batches = 0

        with torch.no_grad():
            for val_features, val_effects in val_dataloader:
                val_features = val_features.to(device)
                val_effects = val_effects.to(device)
                val_pred = adapter(val_features)
                val_loss = F.mse_loss(val_pred, val_effects)
                val_total_loss += val_loss.item()
                val_num_batches += 1

        scheduler.step()
        avg_val_loss = val_total_loss / val_num_batches

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    return adapter

# %%
if __name__ == "__main__":
    adapter = train(15, lr=2e-4)
    
    # Save the trained adapter
    model_name = "9b" if BIG_MODEL else "2b"
    adapter_name = f"adapter_{model_name}_layer_12.pt"
    torch.save(adapter.state_dict(), adapter_name)
    print(f"Adapter saved to {adapter_name}")

    # Calculate and save rotation matrix
    R_dec = calculate_rotation_matrix(adapter, sae)
    R_dec_name = f"R_dec_{model_name}_layer_12.pt"
    torch.save(R_dec, R_dec_name)
    print(f"Rotation matrix saved to {R_dec_name}")

    # Calculate and save correction bias
    correction_bias = calculate_correction_bias(adapter)
    bias_name = f"correction_bias_{model_name}_layer_12.pt"
    torch.save(correction_bias, bias_name)
    print(f"Correction bias saved to {bias_name}")

# %%