import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# Importing the dataset class you already wrote
from dataset import INSAT3D_MultiChannel_Dataset

def visualize_sample():
    # 1. Setup paths
    TIR1_DIR = "data/INSAT3D_TIR1_India"
    VIS_DIR = "data/INSAT3D_VIS_India"
    
    # 2. Initialize dataset
    dataset = INSAT3D_MultiChannel_Dataset(tir1_dir=TIR1_DIR, vis_dir=VIS_DIR, seq_length=12)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 3. Get one batch
    context, target = next(iter(dataloader))
    
    # context shape: (Batch, Time, Channel, Height, Width) -> (1, 6, 2, 128, 128)
    # We'll take the first frame (index 0) of the first batch
    frame = context[0, 0] # Shape: (2, 128, 128)
    
    tir_channel = frame[0].numpy() # TIR1
    vis_channel = frame[1].numpy() # VIS
    
    # 4. Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot TIR1 (Thermal)
    im1 = ax[0].imshow(tir_channel, cmap='inferno')
    ax[0].set_title("TIR1 (Thermal Infrared)")
    plt.colorbar(im1, ax=ax[0])
    
    # Plot VIS (Visible)
    im2 = ax[1].imshow(vis_channel, cmap='gray')
    ax[1].set_title("VIS (Visible)")
    plt.colorbar(im2, ax=ax[1])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_sample()