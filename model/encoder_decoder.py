from torch import nn

def vae_roi_encoder(input_features: int, output_features: int):
    """Build a variational autoencoder (VAE) ROI brain encoder."""
    # Build the encoder.
    encoder = nn.Sequential(
        nn.Linear(input_features, 1024),
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
        nn.Linear(128, output_features),
    )
    return encoder

def vae_roi_decoder(input_features: int, output_features: int):
    """Build a variational autoencoder (VAE) ROI brain decoder."""
    # Build the decoder.
    decoder = nn.Sequential(
        nn.Linear(input_features, 128),
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
        nn.Linear(1024, output_features),
    )
    return decoder


# for image based data, use pretrained model encoder


def vae_img_decoder(input_channels: int, output_channels: int):
    # input is input_channels x 1 x 1
    decoder = nn.Sequential(
        nn.Unflatten(1, (input_channels, 1, 1)),
        nn.ConvTranspose2d(input_channels, 512, 4), # 512 x 4 x 4
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
        nn.Conv2d(32, output_channels, 3, 1, 1), # 32 x 224 x 224,
    )
    decoder.input_channels = input_channels
    decoder.output_channels = output_channels
    return decoder