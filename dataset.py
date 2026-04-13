import os
import glob
import numpy as np
import rioxarray as rxr 
import torch
from torch.utils.data import Dataset, DataLoader

class INSAT3D_MultiChannel_Dataset(Dataset):
    def __init__(self, tir1_dir, vis_dir, seq_length=12):
        """
        tir1_dir: Path to the INSAT3D_TIR1_India folder
        vis_dir: Path to the INSAT3D_VIS_India folder
        """
        self.seq_length = seq_length
        
        # 1. Find and sort files from both directories
        self.tir1_files = sorted(glob.glob(os.path.join(tir1_dir, '*.tif')))
        self.vis_files = sorted(glob.glob(os.path.join(vis_dir, '*.tif')))
        
        # Ensure we have the exact same number of files in both folders
        assert len(self.tir1_files) == len(self.vis_files), "Mismatch in number of TIR1 and VIS files!"
        print(f"Found {len(self.tir1_files)} matching image pairs.")
        
        # 2. Hardcode the Z-Score statistics you already calculated!
        self.tir1_mean, self.tir1_std = 648.81, 112.12
        self.vis_mean, self.vis_std = 76.71, 68.61
        print("Loaded Z-score normalization statistics.")

    def __len__(self):
        return len(self.tir1_files) - self.seq_length + 1

    def __getitem__(self, idx):
        sequence = []
        
        for i in range(idx, idx + self.seq_length):
            # Read both GeoTIFFs
            tir1_img = rxr.open_rasterio(self.tir1_files[i]).squeeze().values
            vis_img = rxr.open_rasterio(self.vis_files[i]).squeeze().values
            
            # Crop the images to 128x128 so it fits on your laptop GPU!
            # We take a patch from the center of the 984x1074 image
            tir1_crop = tir1_img[428:556, 473:601]
            vis_crop = vis_img[428:556, 473:601]
            
            # Apply Independent Z-score normalization
            tir1_norm = (tir1_crop - self.tir1_mean) / self.tir1_std
            vis_norm = (vis_crop - self.vis_mean) / self.vis_std
            
            # Stack them together along the channel axis (Channel 0: TIR1, Channel 1: VIS)
            # Shape becomes (2, 128, 128)
            combined_frame = np.stack([tir1_norm, vis_norm], axis=0)
            sequence.append(combined_frame)
        
        # Convert the full sequence list to a PyTorch Tensor
        # Shape: (Time, Channel, Height, Width) -> (12, 2, 128, 128)
        seq_tensor = torch.tensor(np.array(sequence), dtype=torch.float32)
        
        # Split into Context (Past 6 frames) and Target (Future 6 frames)
        half = self.seq_length // 2
        context = seq_tensor[:half]  
        target = seq_tensor[half:]   
        
        return context, target

# ==========================================
# Testing the DataLoader
# ==========================================
if __name__ == "__main__":
    # Replace these with your actual folder paths
    TIR1_DIR = "data/INSAT3D_TIR1_India" 
    VIS_DIR = "data/INSAT3D_VIS_India"
    
    try:
        dataset = INSAT3D_MultiChannel_Dataset(tir1_dir=TIR1_DIR, vis_dir=VIS_DIR, seq_length=12)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        for context, target in dataloader:
            print("\nSuccess! Multi-modal batch extracted.")
            print(f"Context shape: {context.shape} -> (Batch, Time, Channel, Height, Width)")
            print(f"Target shape: {target.shape} -> (Batch, Time, Channel, Height, Width)")
            break
            
    except Exception as e:
        print(f"Setup Error: {e}")