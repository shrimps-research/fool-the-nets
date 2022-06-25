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
  models_info_by_name,
  dataloader,
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  model_info_by_name = {}
  accuracy_per_model = {}
  confidences_per_model = {}
  for name, model_info in models_info_by_name.items():
    model = model_info.model
    model.eval()
    model_info_by_name[name] = model_info
    accuracy_per_model[name] = load_metric("accuracy")
    confidences_per_model[name] = np.array([])

  for batch in tqdm(dataloader):
    inputs = batch[0].to(device)
    labels = batch[1].to(device)

    for name, model_info in model_info_by_name.items():
      if model_info.requires_normalization:
        inputs = normalize(inputs)

      model = model_info.model
      logits = model(inputs)
      predictions = logits.argmax(-1).cpu().detach().numpy()
      references = labels.cpu().detach().numpy()
      accuracy_per_model[name].add_batch(predictions=predictions, references=references)

      probs = F.softmax(logits, dim=1)
      batch_confidences = probs.gather(1, labels.unsqueeze(1)).cpu().detach().numpy()
      confidences_per_model[name] = np.concatenate((confidences_per_model[name], np.squeeze(batch_confidences)))


  return { name: EvaluationResults(
    accuracy=accuracy_per_model[name].compute()['accuracy'],
    confidence=np.mean(confidences_per_model[name]),
    confidence_std=np.std(confidences_per_model[name])
  ) for name in models_info_by_name.keys()}

