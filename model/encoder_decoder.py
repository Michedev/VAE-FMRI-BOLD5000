from torch import nn

def vae_roi_encoder(input_size: int, output_size: int):
    """Build a variational autoencoder (VAE) ROI brain encoder."""
    # Build the encoder.
    encoder = nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.LayerNorm(1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, output_size),
    )
    return encoder

def vae_roi_decoder(input_size: int, output_size: int):
    """Build a variational autoencoder (VAE) ROI brain decoder."""
    # Build the decoder.
    decoder = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.LayerNorm(1024),
        nn.ReLU(),
        nn.Linear(1024, output_size),
    )
    return decoder


# for image based data, use pretrained model encoder

class ResidualModule(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class VAEBroadcastDecoder(nn.Module):

    def __init__(self, input_size: int, output_size: int,
                 width: int, height: int):
        super().__init__()
        self.width = width
        self.height = height
        self.input_size = input_size
        self.output_size = output_size
        
        self.decoder = nn.Sequential(
            nn.Conv2d(input_size, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResidualModule(16, 16, 3, 1, 1),
            ResidualModule(16, 16, 3, 1, 1),
            nn.Conv2d(16, output_size, 3, 1)
        )


    def forward(self, z):
        z = z.view(*z.shape, 1, 1)
        # broadcast
        z = z.expand(-1, -1, self.width, self.height)
        x_hat = self.decoder(z)
        return x_hat
        
    

def vae_img_decoder(input_size: int, output_size: int):
    # input is bs x input_size
    decoder = nn.Sequential(
        nn.Unflatten(1, (input_size, 1, 1)),
        nn.ConvTranspose2d(input_size, 512, 4), # 512 x 4 x 4
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.ConvTranspose2d(512, 128, 5), # 128 x 8 x 8
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.UpsamplingBilinear2d(scale_factor=4), # 128 x 32 x 32
        nn.ConvTranspose2d(128, 64, 5), # 64 x 36 x 36
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 5), # 32 x 40 x 40
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.UpsamplingBilinear2d(scale_factor=3), # 32 x 120 x 120
        nn.Conv2d(32, 32, 5), # 32 x 116 x 116
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 5), # 32 x 112 x 112
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.UpsamplingBilinear2d(scale_factor=2), # 32 x 224 x 224
        nn.Conv2d(32, 32, 3, 1, 1), # 32 x 224 x 224
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, output_size, 3, 1, 1), # 32 x 224 x 224,
    )
    decoder.input_size = input_size
    decoder.output_size = output_size
    return decoder