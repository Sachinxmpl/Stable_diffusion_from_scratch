import torch 
from torch import nn 
from torch.nn import functional as F 
from .attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self , in_channels , out_channels):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels , out_channels , kernel_size = 3 , padding = 1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels , out_channels , kernel_size = 3 , padding = 1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels , out_channels, kernel_size = 1 , padding = 0)

    def forward(self, x):
            residue = x 
            
            x = self.norm1(x)
            x = F.silu(x)
            x = self.conv1(x)
            x = self.norm2(x)
            x = F.silu(x)
            x = self.conv2(x)

            return x  + self.residual_layer(residue)


class VAE_AttentionBlock(nn.Module):
    def __init__(self , channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32 , channels)
        self.attention = SelfAttention(1 , channels)

    def forward(self , x):
        residue = x
        x = self.groupnorm(x)

        b , c , h , w = x.shape
        x = x.view((b, c, h*w)).transpose(-1, -2)  # (B, H*W, C)
        x = self.attention(x)
        x = x.transpose(-1, -2).view((b, c, h, w))

        x += residue

        return x 
    

class VAE_Encoder(nn.Module):
    def __init__(self , latent_channels = 4 ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3 , 128 , kernel_size = 3 , padding = 1),
            VAE_ResidualBlock(128 , 128) , 
            VAE_ResidualBlock(128 , 128) , 
            nn.Conv2d(128 , 128 , kernel_size = 3 , stride = 2 , padding = 1),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),

            # latent_channels * 2 beacause vae each latent channels requires two components mean(center of distribution) , log variance 
            nn.Conv2d(512, latent_channels * 2, kernel_size=3, padding=1)
        )

    def forward(self , x, noise = None):
        x = self.layers(x)

        # (B , latent_channles * 2 = 8 , H / 8 , W / 8) -> two tensors of shape (B , 4 , H/8 , W/8)
        mean , logvar = torch.chunk(x , 2 , dim = 1)

        # clamps the variance between -30 and 20
        logvar = torch.clamp(logvar , -30 , 20)
        variance = logvar.exp()
        std = variance.sqrt()

        if noise is None:
            noise = torch.randn_like(std)

        # VAE cannot backpropagate through random sampling directly.
        # Trick: sample ε ~ N(0,1) and compute z = μ + σ * ε
        # This allows gradients to flow through μ and σ during training.

        z = mean + std * noise
        z = z * 0.18215

        return z , mean , logvar
        
class VAE_Decoder(nn.Module):
    def __init__(self, latent_channels=4, final_activation='sigmoid'):
        super().__init__()
        self.final_activation = final_activation
        self.layers = nn.Sequential(
            nn.Conv2d(latent_channels , latent_channels , kernel_size=1),
            nn.Conv2d(latent_channels, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),


            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = x / 0.18215  
        x = self.layers(x)
        if self.final_activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x