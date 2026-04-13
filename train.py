import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the modules we built in the previous steps
from dataset import INSAT3D_MultiChannel_Dataset
from unet_3d import SpatiotemporalUNet
from diffusion import GaussianDiffusion

def train_model():
    # 1. Setup Data Paths (Replace with your actual paths)
    TIR1_DIR = "data/INSAT3D_TIR1_India" 
    VIS_DIR = "data/INSAT3D_VIS_India"
    
    print("Loading datasets...")
    dataset = INSAT3D_MultiChannel_Dataset(tir1_dir=TIR1_DIR, vis_dir=VIS_DIR, seq_length=12)
    # Batch size of 2 keeps memory usage low for your laptop
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True) 

    # 2. Initialize Models
    print("Initializing architecture...")
    # Determine if a compatible GPU is available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    unet = SpatiotemporalUNet(in_channels=2, out_channels=2).to(device)
    diffusion_model = GaussianDiffusion(unet, timesteps=1000).to(device)

    # 3. Setup Optimizer
    # Adam is the standard optimizer for diffusion models
    optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-4)

    # 4. Training Loop
    epochs = 5 # Small number for a dummy test run
    
    print("Starting training loop...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (context, target) in enumerate(dataloader):
            # Move data to GPU and swap Time/Channel to match (Batch, Channel, Time, Height, Width)
            context = context.to(device).permute(0, 2, 1, 3, 4)
            target = target.to(device).permute(0, 2, 1, 3, 4)

            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass: Calculate noise prediction loss
            loss = diffusion_model(context, target)
            
            # Backward pass: Calculate gradients
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ---")

    print("Dummy training run complete! Your model is learning.")
    
    return diffusion_model

if __name__ == "__main__":
    diffusion_model = train_model()
    # Save the trained model weights to your hard drive
    torch.save(diffusion_model.state_dict(), "insat_diffusion_model.pth")
    print("Model successfully saved to disk!")  