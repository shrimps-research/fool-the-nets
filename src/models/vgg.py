import torch
from src.models.common import PretrainedModel
from torchvision.models import vgg16
VGG = 'vgg'

VGG_PRETRAINED_MODELS = {
  VGG: PretrainedModel(
    weights_uri=None,
    feature_extractor=None,
    expected_image_size=224,
    model=vgg16(pretrained=True),
  ),
}

def get_vgg(
  type=VGG,
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
  pretrained_model:PretrainedModel = VGG_PRETRAINED_MODELS[type]
  pretrained_model.model.to(device)
  return pretrained_model
