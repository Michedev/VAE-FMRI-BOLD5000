import torch
from model.encoder_decoder import vae_roi_encoder, vae_roi_decoder, vae_img_decoder


@torch.no_grad()
def test_decoder_img():
    input_size = 1000
    output_size = 3
    decoder = vae_img_decoder(input_size, output_size)
    nn_output = decoder(torch.rand(1, input_size))
    assert nn_output.shape == (1, 3, 224, 224)


@torch.no_grad()
def test_vae_encoder():
    encoder = vae_roi_encoder(1000, 100)
    nn_output = encoder(torch.rand(1, 1000))
    assert nn_output.shape == (1, 100)


@torch.no_grad()
def test_vae_decoder():
    decoder = vae_roi_decoder(50, 1000)
    nn_output = decoder(torch.rand(1, 50))
    assert nn_output.shape == (1, 1000)
