from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_metric
from torch.autograd import Variable
from tqdm import tqdm

@dataclass
class PretrainedModel:
  weights_uri: str
  feature_extractor: object
  expected_image_size: int
  model: object


@dataclass
class EvaluationResults:
  accuracy: float
  confidence: float

def evaluate(
  model,
  dataloader,
):
  model.eval()
  accuracy = load_metric("accuracy")
  confidences = np.array([])

  for batch in tqdm(dataloader):

    inputs = batch[0]
    labels = batch[1]

    logits = model(inputs)
    predictions = logits.argmax(-1).cpu().detach().numpy()
    references = labels.numpy()
    accuracy.add_batch(predictions=predictions, references=references)

    probs = F.softmax(logits, dim=1)
    batch_confidences = probs.gather(1, labels.unsqueeze(1)).cpu().detach().numpy()
    confidences = np.concatenate((confidences, np.squeeze(batch_confidences)))

  return EvaluationResults(
    accuracy=accuracy.compute()['accuracy'],
    confidence=np.mean(confidences),
  )

