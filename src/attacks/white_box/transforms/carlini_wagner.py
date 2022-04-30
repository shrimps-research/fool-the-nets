import torch
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

class CarliniWagnerL2(torch.nn.Module):

  def __init__(
    self,
    model,
    target=None,
    binary_search_steps=5,
    max_iterations=2,
    n_classes=1000 # pretrained models in imagenet1k
  ):
    super().__init__()
    self.model = model
    self.target = target
    self.binary_search_steps = binary_search_steps
    self.max_iterations = max_iterations
    self.n_classes=n_classes

  def forward(self, img):
    if self.target is None:
      image_with_noise = carlini_wagner_l2(
        model_fn=self.model,
        x=img.unsqueeze(0),
        n_classes=self.n_classes,
        max_iterations=self.max_iterations

      )
    else:
      image_with_noise = carlini_wagner_l2(
        model_fn=self.model,
        x=img.unsqueeze(0),
        n_classes=self.n_classes,
        max_iterations=self.max_iterations,
        y=self.target,
        targeted=True,
      )
    # print(image_with_noise.squeeze().shape)
    return image_with_noise.squeeze()


  def __repr__(self) -> str:
    return f"{self.__class__.__name__}"


