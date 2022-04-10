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
  confidences: np.ndarray

def evaluate(
  model,
  dataloader,
  feature_extractor=None,
  preprocessing_functions=[],
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
  # model.eval()
  accuracy = load_metric("accuracy")
  confidences = torch.tensor([])

  for batch in tqdm(dataloader):

    if feature_extractor:
      batch_array = Variable(batch[0], requires_grad=True).detach().numpy()
      inputs = feature_extractor(
        list(batch_array),
        return_tensors="pt"
      ).pixel_values.to(device)
    else:
      inputs = Variable(batch[0], requires_grad=True)


    for f in preprocessing_functions:
      inputs = f(inputs)


    logits = model(inputs)
    predictions = logits.argmax(-1).cpu().detach().numpy()
    references = batch[1].numpy()
    probs = F.softmax(logits, dim=1)
    batch_confidences = probs.gather(1, batch[1].unsqueeze(1))
    confidences = torch.cat([confidences, batch_confidences])

    accuracy.add_batch(predictions=predictions, references=references)


  return EvaluationResults(
    accuracy=accuracy.compute()['accuracy'],
    confidences=confidences.cpu().detach().numpy(),
  )

