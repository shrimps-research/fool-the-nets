import torch
from transformers import PerceiverFeatureExtractor, PerceiverConfig, PerceiverForImageClassificationLearned

from src.data.imagenet_labels_mapping import imagenet_class_idx_mappings
from src.models.common import PretrainedModel

PERCEIVER_IO_LEARNED_POS_EMBEDDINGS = 'perceiver-io'
PERCEIVER_IO_FOURIER= 'perceiver-io-fourier'

PRETRAINED_MODELS = {
  PERCEIVER_IO_LEARNED_POS_EMBEDDINGS: PretrainedModel(
    weights_uri="deepmind/vision-perceiver-learned",
    feature_extractor=PerceiverFeatureExtractor(),
    expected_image_size=224,
    model=None,
    requires_normalization=True,
  ),
  # TODO: Fourier model has not been tested yet. Expected image size to be defined as well
  PERCEIVER_IO_FOURIER: PretrainedModel(
    weights_uri="deepmind/vision-perceiver-fourier",
    feature_extractor=PerceiverFeatureExtractor.from_pretrained("deepmind/vision-perceiver-fourier"),
    expected_image_size=224,
    model=None,
    requires_normalization=True,
  ),
}

class PerceiverIO(torch.nn.Module):

  def __init__(self, weights_uri):
    super(PerceiverIO, self).__init__()

    class_to_idx, idx_to_class = imagenet_class_idx_mappings()
    config, unused_kwargs = PerceiverConfig.from_pretrained(
      weights_uri,
      return_dict=False,
      id2label=class_to_idx,
      label2id=idx_to_class,
      return_unused_kwargs=True
    )

    if unused_kwargs:
      print(f'Unused argument exist in PerceiverConfig: {unused_kwargs}')

    self.model = PerceiverForImageClassificationLearned.from_pretrained(
      weights_uri,
      config=config,
      # ignore_mismatched_sizes=True,
    )

  def forward(self, img_batch):
    return self.model(img_batch)[0]


def get_perceiver_io(
  type=PERCEIVER_IO_LEARNED_POS_EMBEDDINGS,
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):

  pretrained_perceiver_io: PretrainedModel = PRETRAINED_MODELS[type]
  model = PerceiverIO(pretrained_perceiver_io.weights_uri)
  model.to(device)
  pretrained_perceiver_io.model = model
  return pretrained_perceiver_io
