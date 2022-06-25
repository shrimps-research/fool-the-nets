from torch import nn
from src.models.vit import get_vit, ViT

class MyConvTranspose2d(nn.Module):
  def __init__(self, conv, output_size):
    super(MyConvTranspose2d, self).__init__()
    self.output_size = output_size
    self.conv = conv

  def forward(self, x):
    x = self.conv(x, output_size=self.output_size)
    return x

class Reshape(nn.Module):
  def __init__(self, *args):
    super(Reshape, self).__init__()
    self.shape = args

  def forward(self, x):
    return x.view(self.shape)

class Repeat(nn.Module):
  def __init__(self, *repeat_times):
    super(Repeat, self).__init__()
    self.repeat_times = repeat_times

  def forward(self, x):
    return x.repeat(self.repeat_times)


class PretrainedAutoencoder(nn.Module):
  def __init__(self,):
    super(PretrainedAutoencoder, self).__init__()
    self.encoder = get_vit(ViT, num_classes=0).model
    self.encoder.requires_grad = False
    self.upsample = nn.Sequential(
      nn.Linear(768, 28*28),
      nn.SELU(),
      Reshape(-1, 1, 28, 28),
      nn.ConvTranspose2d(1, 16, 3, stride=2, padding=1, output_padding=1),  # -> N, 16, 56, 56
      nn.SELU(),
      nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # -> N, 8, 112, 112
      nn.SELU(),
      nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1),  # -> N, 3, 224, 224
      nn.Sigmoid()
    )


  def forward(self, input_images):
    embedding = self.encoder(input_images)
    output_img = self.upsample(embedding)
    return output_img

