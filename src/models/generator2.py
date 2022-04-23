import torch
from torch import nn
from torchsummary import summary
import warnings

warnings.filterwarnings("ignore")

class MaxPoolUnpool2D(nn.Module):

  def __init__(self, kernel, stride=2):
    super(MaxPoolUnpool2D, self).__init__()
    self.pool = nn.MaxPool2d(kernel, stride, return_indices=True, ceil_mode=False)
    self.unpool = nn.MaxUnpool2d(kernel, stride)
    self.indices = None

  def forward(self, x):
    if self.indices is None:
      output, self.indices = self.pool.forward(x)
    else:
      output = self.unpool.forward(x, self.indices)
      self.indices = None

    return output

class NoiseGenerator(nn.Module):
  def __init__(self, image_size=224 * 224):
    super(NoiseGenerator, self).__init__()
    self.pool_unpool_1 = MaxPoolUnpool2D(2)
    self.pool_unpool_2 = MaxPoolUnpool2D(2)

    self.encoder = nn.Sequential(
      nn.Conv2d(3, 16, 3, stride=2, padding=1),  # -> N, 16, 112, 112
      nn.SELU(),
      self.pool_unpool_1,  # -> N, 56, 56
      # nn.MaxPool2d(2, 2),
      nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> N, 32, 28, 28
      nn.SELU(),
      nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> N, 64, 14, 14
      nn.SELU(),
    )

    # N , 64, 1, 1
    self.decoder = nn.Sequential(
      # nn.ConvTranspose2d(128, 64, 7),  # -> N, 64, 7, 7
      # nn.SELU(),
      nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # -> N, 32, 28, 28
      nn.SELU(),
      # self.pool_unpool_2,               # -> N, 64, 14, 14
      nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # -> N, 16, 112, 112
      nn.SELU(),
      self.pool_unpool_1,  # -> N, 32, 56, 56
      nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # N, 3, 224, 224
      # nn.SELU()
      # nn.Linear()
      # nn.Hardtanh(min_val=-5, max_val=5)
      nn.Hardtanh(min_val=-1, max_val=1)  # -> used for noisy images generation
    )

    self.model = nn.Sequential(
      self.encoder,
      self.decoder
    )
    summary(self.model, (3, 224, 224))

  def forward(self, input_image):
    # encoded = self.encoder_cnn(input_image)
    # noise = self.decoder_conv(encoded)
    noise = 0.2 * self.model(input_image)
    # noise = torch.clamp(self.model(input_image), min=-0.2, max=0.2)
    return noise
    # adversarial_image = input_image + noise
    # return adversarial_image
