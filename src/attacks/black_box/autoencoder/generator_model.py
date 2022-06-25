import torch
from torch import nn

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
  def __init__(self):
    super(NoiseGenerator, self).__init__()
    self.pool_unpool_1 = MaxPoolUnpool2D(2)
    self.pool_unpool_2 = MaxPoolUnpool2D(2)
    self.noise_flag = False
    self.selu = nn.SELU()
    self.conv_1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
    self.conv_2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
    self.conv_3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
    self.conv_11 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
    self.conv_22 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
    self.conv_33 = nn.Conv2d(64, 3, 3, stride=2, padding=1)
    self.deconv_1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
    self.deconv_2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
    self.deconv_3 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)
    self.sigmoid = nn.Sigmoid()
    self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1)

  def forward(self, input_images):
    x = input_images
    x = self.selu(self.conv_1(x))
    x = self.deconv_3(x)

    return x

  def forward2(self, input_image):
    output_image = self.model(input_image)
    if self.noise_flag:
      noise = torch.clamp(output_image - input_image, -0.08, 0.08)
      output_image = input_image + noise

    return output_image

  def enable_noise(self):
    self.conv_1.requires_grad_(False)
    self.conv_2.requires_grad_(False)
    self.conv_3.requires_grad_(False)
    self.deconv_1.requires_grad_(False)
    self.deconv_2.requires_grad_(False)
