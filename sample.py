import torch
import matplotlib.pyplot as plt
from unet_3d import SpatiotemporalUNet
from diffusion import GaussianDiffusion
from dataset import INSAT3D_MultiChannel_Dataset
from torch.utils.data import DataLoader

@torch.no_grad() # We don't need gradients for inference
def generate_forecast():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize the architecture and load your saved weights
    print("Loading trained model...")
    unet = SpatiotemporalUNet(in_channels=2, out_channels=2).to(device)
    diffusion_model = GaussianDiffusion(unet, timesteps=1000).to(device)
    
    # Load the.pth file you just created!
    # (Setting strict=False just in case there are minor buffer mismatches)
    diffusion_model.load_state_dict(torch.load("insat_diffusion_model.pth", map_location=device), strict=False)
    diffusion_model.eval()

    # 2. Get a batch of past context frames
    TIR1_DIR = "data/INSAT3D_TIR1_India" 
    VIS_DIR = "data/INSAT3D_VIS_India"
    dataset = INSAT3D_MultiChannel_Dataset(tir1_dir=TIR1_DIR, vis_dir=VIS_DIR, seq_length=12)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    context, actual_future = next(iter(dataloader))
    context = context.to(device).permute(0, 2, 1, 3, 4) # Shape: (1, 2, 6, 128, 128)
    
    print("Starting reverse diffusion process (this may take a minute on CPU)...")
    
    # 3. Start with pure random noise for the future frames
    generated_future = torch.randn_like(context)
    
    # 4. The Reverse Loop: Step backwards from T=1000 to 0
    for t in reversed(range(0, diffusion_model.timesteps)):
        # Create a tensor for the current timestep
        t_batch = torch.tensor([t], device=device).long()
        
        # Combine the clean past with the currently noisy future
        model_input = torch.cat([context, generated_future], dim=2)
        
        # Predict the noise
        predicted_noise = diffusion_model.model(model_input)[:, :, 6:, :, :]
        
        # Get the alpha/beta values for this timestep to scale the subtraction
        alpha = (1.0 - torch.linspace(1e-4, 0.02, 1000).to(device))[t]
        alpha_cumprod = diffusion_model.alpha_cumprod[t]
        beta = 1.0 - alpha
        
        # Remove a fraction of the predicted noise
        if t > 0:
            noise = torch.randn_like(generated_future)
        else:
            noise = torch.zeros_like(generated_future) # No noise added on the very last step
            
        generated_future = (1 / torch.sqrt(alpha)) * (generated_future - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
        
        if t % 100 == 0:
            print(f"Sampling step {t}/1000 completed.")

    print("Forecast generated!")
    
    # 5. Visualize the result (Comparing the first generated frame to the real one)
    # Extract the TIR1 channel (index 0) of the first forecasted frame (index 0)
    pred_img = generated_future[0, 0, 0, :, :].cpu().numpy()
    real_img = actual_future[0, 0, 0, :, :].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes.imshow(real_img, cmap='gray')
    axes.set_title("Actual Future Cloud")
    axes[1].imshow(pred_img, cmap='gray')
    axes[1].set_title("AI Generated Forecast")
    plt.show()

if __name__ == "__main__":
    generate_forecast()