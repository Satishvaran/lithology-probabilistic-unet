import torch

from lithology_unet.models import ProbabilisticUNet1D, UNet1D


def test_unet_shapes():
    x = torch.randn(2, 12, 128)
    assert UNet1D(12, 12, base=8)(x).shape == (2, 12, 128)


def test_probabilistic_unet_shapes():
    x = torch.randn(2, 12, 128)
    y = torch.randint(0, 12, (2, 128))
    model = ProbabilisticUNet1D(12, 12, base=8, latent_dim=4)
    logits, prior, posterior = model(x, y)
    assert logits.shape == (2, 12, 128)
    assert prior[0].shape == posterior[0].shape == (2, 4)
    assert model.sample(x, n=3).shape == (3, 2, 12, 128)
