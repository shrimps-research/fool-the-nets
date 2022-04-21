import torch
import numpy as np
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

class FastGradientTransform(torch.nn.Module):

  def __init__(self, model, target=None, epsilon=0.03, norm=np.inf):
    super().__init__()
    self.model = model
    self.epsilon = epsilon
    self.norm = norm
    self.target = target

  def forward(self, img):
    if self.target is None:
      image_with_noise = fast_gradient_method(
        self.model,
        img.unsqueeze(0),
        self.epsilon,
        self.norm
      )
    else:
      image_with_noise = fast_gradient_method(
        self.model,
        img.unsqueeze(0),
        self.epsilon,
        self.norm,
        y=self.target,
        targeted=True
      )

    return image_with_noise.squeeze()


  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(epsilon={self.epsilon})"


