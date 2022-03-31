from collections import namedtuple

import torch
from datasets import load_metric
from tqdm import tqdm
from transformers import PerceiverFeatureExtractor

VISION_LEARNED_POS_EMBEDDINGS = 'vision perceiver learned'
VISION_FOURIER = 'deepmind/vision-perceiver-fourier'

PretrainedModel = namedtuple(
    "PretrainedModel",
    field_names=["weights", "feature_extractor", "expected_image_size"],
)

PRETRAINED_MODELS = {
  VISION_LEARNED_POS_EMBEDDINGS: PretrainedModel(
    weights="deepmind/vision-perceiver-learned",
    feature_extractor=PerceiverFeatureExtractor(),
    expected_image_size=224,
  ),
  # TODO: Fourier model has not been tested yet. Expected image size to be defined as well
  VISION_FOURIER: PretrainedModel(
    weights="deepmind/vision-perceiver-fourier",
    feature_extractor=PerceiverFeatureExtractor.from_pretrained("deepmind/vision-perceiver-fourier"),
    # expected_image_size=224,
  ),
}

def evaluate(model, feature_extractor, dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
  accuracy = load_metric("accuracy")
  model.eval()
  for batch in tqdm(dataloader):
    # get the inputs;
    inputs = feature_extractor(list(batch[0]), return_tensors="pt").pixel_values.to(device)
    labels = batch[1].to(device)

    # forward pass
    outputs = model(inputs=inputs, labels=labels)
    logits = outputs.logits
    predictions = logits.argmax(-1).cpu().detach().numpy()
    references = batch[1].numpy()
    accuracy.add_batch(predictions=predictions, references=references)

  return accuracy.compute()