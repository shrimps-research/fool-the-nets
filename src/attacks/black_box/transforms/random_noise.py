import torch

class GaussianNoiseTransform(torch.nn.Module):

  def __init__(self, model, epsilon=0.03):
    super().__init__()
    self.model = model
    self.epsilon = epsilon
    self.mean = 0
    self.std = 1

  def forward(self, img):
    noise = torch.randn(img.size()) * self.std + self.mean
    noise = torch.clamp(noise, min=-self.epsilon, max=self.epsilon)
    return img + noise


  def __repr__(self) -> str:
    return f"{self.__class__.__name__}"


