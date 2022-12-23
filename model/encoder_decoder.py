from torch import nn

def vae_roi_encoder(input_features: int, output_features: int):
    """Build a variational autoencoder (VAE) ROI brain encoder."""
    # Build the encoder.
    encoder = nn.Sequential(
        nn.Linear(input_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, output_features),
    )
    return encoder

def vae_roi_decoder(input_features: int, output_features: int):
    """Build a variational autoencoder (VAE) ROI brain decoder."""
    # Build the decoder.
    decoder = nn.Sequential(
        nn.Linear(input_features, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, output_features),
    )
    return decoder


# for image based data, use pretrained model encoder