from dataclasses import dataclass
from torchvision import transforms as T
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_metric
from tqdm import tqdm

@dataclass
class PretrainedModel:
  weights_uri: str
  feature_extractor: object
  expected_image_size: int
  model: torch.nn.Module
  requires_normalization: bool


@dataclass
class EvaluationResults:
  accuracy: float
  confidence: float
  confidence_std: float

def evaluate(
  model_info: PretrainedModel,
  dataloader,
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  model = model_info.model
  # model = torch.nn.DataParallel(model, device_ids=[0, 1])
  model.eval()
  accuracy = load_metric("accuracy")
  confidences = np.array([])

  for batch in tqdm(dataloader):

    inputs = batch[0].to(device)
    if model_info.requires_normalization:
      inputs = normalize(inputs)

    labels = batch[1].to(device)

    logits = model(inputs)
    predictions = logits.argmax(-1).cpu().detach().numpy()
    references = labels.cpu().detach().numpy()
    accuracy.add_batch(predictions=predictions, references=references)

    probs = F.softmax(logits, dim=1)
    batch_confidences = probs.gather(1, labels.unsqueeze(1)).cpu().detach().numpy()
    confidences = np.concatenate((confidences, np.squeeze(batch_confidences)))

  return EvaluationResults(
    accuracy=accuracy.compute()['accuracy'],
    confidence=np.mean(confidences),
    confidence_std=np.std(confidences)
  )

