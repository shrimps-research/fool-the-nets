import torch
from transformers import AutoFeatureExtractor, SwinForImageClassification, SwinConfig


from src.data.imagenet_labels_mapping import imagenet_class_idx_mappings
from src.models.common import PretrainedModel

SWIN = 'swin'

PRETRAINED_MODELS = {
  SWIN: PretrainedModel(
    weights_uri="microsoft/swin-tiny-patch4-window7-224",
    feature_extractor=AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224"),
    expected_image_size=224,
    model=None
  ),
}

class Swin(torch.nn.Module):

  def __init__(self, weights_uri):
    super(Swin, self).__init__()

    class_to_idx, idx_to_class = imagenet_class_idx_mappings()
    config, unused_kwargs = SwinConfig.from_pretrained(
      weights_uri,
      return_dict=False,
      id2label=class_to_idx,
      label2id=idx_to_class,
      return_unused_kwargs=True
    )

    if unused_kwargs:
      print(f'Unused argument exist in SwinConfig: {unused_kwargs}')

    self.model = SwinForImageClassification.from_pretrained(
      weights_uri,
      config=config,
      # ignore_mismatched_sizes=True,
    )

  def forward(self, img_batch):
    return self.model(img_batch)[0]


def get_swin(
  type=SWIN,
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):

  swin: PretrainedModel = PRETRAINED_MODELS[type]
  model = Swin(swin.weights_uri)
  model.to(device)
  swin.model = model
  return swin
