import timm
import torch
from src.models.common import PretrainedModel

XCEPTION = 'xception'

XCEPTION_PRETRAINED_MODELS = {
  XCEPTION: PretrainedModel(
    weights_uri=None,
    feature_extractor=None,
    expected_image_size=224,
    model=timm.create_model('xception', pretrained=True),
    requires_normalization=True,
  ),
}

def get_xception(
  type=XCEPTION,
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
  pretrained_model:PretrainedModel = XCEPTION_PRETRAINED_MODELS[type]
  pretrained_model.model.to(device)
  return pretrained_model
