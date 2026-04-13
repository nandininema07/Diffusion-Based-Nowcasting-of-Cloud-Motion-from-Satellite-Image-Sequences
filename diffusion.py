import torch
import torch.nn as nn

class GaussianDiffusion(nn.Module):
    def __init__(self, unet_model, timesteps=1000):
        super().__init__()
        self.model = unet_model
        self.timesteps = timesteps

        # 1. Define the linear variance schedule (beta)
        # This controls how much noise is added at each step
        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)

        # Register as buffers so PyTorch automatically moves them to the GPU with the model
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1.0 - alpha_cumprod))

    def forward(self, context, target):
        """
        context: The clean past frames (T=6)
        target: The clean future frames we want to predict (T=6)
        """
        B = target.shape[0]  # Batch size

        # 2. Sample a random timestep 't' for each sequence in the batch
        t = torch.randint(0, self.timesteps, (B,), device=target.device).long()

        # 3. Generate pure random Gaussian noise
        noise = torch.randn_like(target)

        # 4. Add the noise to the target frames based on the drawn timestep 't'
        # We reshape the alpha constants to broadcast across (Batch, Channel, Time, Height, Width)
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(B, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t].view(B, 1, 1, 1, 1)
        
        # The noisy future target
        noisy_target = sqrt_alpha * target + sqrt_one_minus_alpha * noise

        # 5. Concatenate the clean past context with the noisy future target along the TIME dimension (dim=2)
        # Context is 6 frames, Noisy Target is 6 frames -> Input to U-Net is 12 frames
        model_input = torch.cat([context, noisy_target], dim=2) 

        # 6. Predict the noise using the 3D U-Net
        predicted_output = self.model(model_input)

        # 7. Extract only the predictions for the future frames (the last 6 frames)
        predicted_noise = predicted_output[:, :, 6:, :, :]

        # 8. Calculate Hybrid Loss (MSE + L1) for sharper cloud boundaries
        mse_loss = nn.functional.mse_loss(predicted_noise, noise)
        l1_loss = nn.functional.l1_loss(predicted_noise, noise)
        
        # Combine them (you can weight the L1 loss, e.g., by 0.5)
        loss = mse_loss + (0.5 * l1_loss)

        return loss

# ==========================================
# Testing the Diffusion Wrapper
# ==========================================
if __name__ == "__main__":
    from unet_3d import SpatiotemporalUNet
    
    print("Initializing U-Net and Diffusion Model...")
    # Initialize our 2-channel U-Net
    unet = SpatiotemporalUNet(in_channels=2, out_channels=2)
    diffusion_model = GaussianDiffusion(unet)
    
    # Create fake batch data mimicking our dataloader output
    # (Batch=2, Channels=2, Time=6, Height=128, Width=128)
    dummy_context = torch.randn((2, 2, 6, 128, 128))
    dummy_target = torch.randn((2, 2, 6, 128, 128))
    
    print("Executing Forward Diffusion and U-Net Noise Prediction...")
    loss = diffusion_model(dummy_context, dummy_target)
    
    print(f"Success! Calculated Noise MSE Loss: {loss.item():.4f}")