import timm
import torch

from src.models.common import PretrainedModel

ViT = 'vit'

ViT_PRETRAINED_MODELS = {
  ViT: PretrainedModel(
    weights_uri="vit_base_patch16_224",
    feature_extractor=None,
    expected_image_size=224,
    model=None,
    requires_normalization=False,
  ),
}

def get_vit(
  type=ViT,
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
  num_classes = 1000
):
  pretrained_model:PretrainedModel = ViT_PRETRAINED_MODELS[type]

  model = timm.create_model(
    pretrained_model.weights_uri,
    pretrained=True,
    num_classes=num_classes
  )
  model.to(device)

  pretrained_model.model = model
  return pretrained_model
