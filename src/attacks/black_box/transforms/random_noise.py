import torch

class GaussianNoiseTransform(torch.nn.Module):

  def __init__(self, model, epsilon=0.03):
    super().__init__()
    self.model = model
    self.epsilon = epsilon
    self.mean = 0
    self.std = 1
    noise = torch.randn((224,224)) * self.std + self.mean
    noise = torch.clip(noise, min=-self.epsilon, max=self.epsilon)
    self.noise = noise

  def forward(self, img):
    # noise = torch.randn(img.size()) * self.std + self.mean
    # noise = torch.clip(noise, min=-self.epsilon, max=self.epsilon)
    noisy_img = torch.clip(img + self.noise, min=0.0, max=1.0)
    return noisy_img


  def __repr__(self) -> str:
    return f"{self.__class__.__name__}"


