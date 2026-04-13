import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    """A standard block consisting of two 3D Convolution layers and Normalization."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            # Padding=1 ensures the spatial and temporal dimensions don't shrink during convolution
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.net(x)

class SpatiotemporalUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # We only want to pool/downsample the Spatial dimensions (Height, Width) 
        # We leave the Temporal dimension (Time) untouched to preserve sequence length
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) 

        # Encoder (Downsampling path)
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Bottleneck (Deepest part of the network)
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        # Decoder (Upsampling path)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )
            self.ups.append(DoubleConv3D(feature * 2, feature))

        # Final Convolution to map back to the target number of channels (1 for TIR1)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Go down the U-Net
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse the list for decoding

        # Go up the U-Net
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            
            # Concatenate the skip connection with the upsampled data along the channel dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)

        return self.final_conv(x)

# ==========================================
# Testing the U-Net
# ==========================================
if __name__ == "__main__":
    # PyTorch 3D Convolutions expect the channel dimension to come BEFORE the time dimension.
    # Shape: (Batch, Channels, Time, Height, Width)
    # We use a 128x128 patch to fit in laptop memory!
    dummy_input = torch.randn((2, 2, 6, 128, 128)) 
    
    print("Initializing 3D U-Net...")
    model = SpatiotemporalUNet(in_channels=2, out_channels=2)
    
    print("Passing data through the model...")
    predictions = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {predictions.shape}")