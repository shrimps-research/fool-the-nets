import torch
import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

class ProjectedGradientTransform(torch.nn.Module):

  def __init__(
    self,
    model,
    target=None,
    epsilon=0.03,
    step_size=0.005,
    iterations=10,
    norm=np.inf
  ):
    super().__init__()
    self.model = model
    self.epsilon = epsilon
    self.step_size = step_size
    self.iterations = iterations
    self.norm = norm
    self.target = target

  def forward(self, img):
    if self.target is None:
      image_with_noise = projected_gradient_descent(
        model_fn=self.model,
        x=img.unsqueeze(0),
        eps=self.epsilon,
        eps_iter=self.step_size,
        nb_iter=self.iterations,
        norm=self.norm,
      )
    else:
      image_with_noise = projected_gradient_descent(
        model_fn=self.model,
        x=img.unsqueeze(0),
        eps=self.epsilon,
        eps_iter=self.step_size,
        nb_iter=self.iterations,
        norm=self.norm,
        y=self.target,
        targeted=True,
      )

    return image_with_noise.squeeze()


  def __repr__(self) -> str:
    return f"{self.__class__.__name__}"


