from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import torch
import numpy as np
import os

from src.data.datasets import IMAGENET100
from src.settings import DEFAULT_DATASETS_PATH
from src.data.imagenet_labels_mapping import imagenet_class_idx_mappings

class ImageData:
  def __init__(self, dataset, dataloader):
    self.dataset = dataset
    self.dataloader = dataloader

def get_dataset_and_dataloader(
  dataset=IMAGENET100,
  dataset_type='train',
  max_size=None,
  batch_size=2,
  transform=T.ToTensor(),
  datasets_path=DEFAULT_DATASETS_PATH,
):
  torch.manual_seed(0xDAB)
  np.random.seed(0xDAB)

  parent_dataset_path = os.path.join(datasets_path, dataset)
  requested_dataset_path = os.path.join(parent_dataset_path, dataset_type)

  if not os.path.isdir(parent_dataset_path):
    raise Exception(f'Dataset {dataset} does not exist under {datasets_path} directory')
  if not os.path.isdir(requested_dataset_path):
    raise Exception(f'{dataset_type} set does not exist under {datasets_path}/{dataset} directory')

  dataset = datasets.ImageFolder(requested_dataset_path, transform=transform)
  dataset = mapped_to_imagenet_indices(dataset)

  if max_size:
    indices = np.random.choice(np.arange(len(dataset.samples)), max_size, replace=False)
    dataset.samples = [dataset.samples[idx] for idx in indices]
    dataset.targets = [dataset.targets[idx] for idx in indices]

  kwargs = {"pin_memory": False, "num_workers": 0} if torch.cuda.is_available() else {}

  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

  return dataset, dataloader


def mapped_to_imagenet_indices(dataset):
  class_to_idx, idx_to_class = imagenet_class_idx_mappings()
  dataset_idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
  targets, samples = [], []
  for i, target in enumerate(dataset.targets):
    class_of_target = dataset_idx_to_class[target]
    class_idx = class_to_idx[class_of_target]
    targets.append(class_idx)
    samples.append((dataset.samples[i][0], class_idx))
  dataset.targets = targets
  dataset.samples = samples
  for key, idx in dataset.class_to_idx.items():
    dataset.class_to_idx[key] = class_to_idx[key]

  return dataset

