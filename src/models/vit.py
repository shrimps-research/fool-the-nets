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
  ),
}

def get_vit(
  type=ViT,
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
  pretrained_model:PretrainedModel = ViT_PRETRAINED_MODELS[type]

  model = timm.create_model(
    pretrained_model.weights_uri,
    pretrained=True
  )
  model.to(device)

  pretrained_model.model = model
  return pretrained_model
