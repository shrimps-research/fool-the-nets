from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import torch
import os

from src.data.datasets import IMAGENET100
from src.settings import DEFAULT_DATASETS_PATH

class ImageData:
  def __init__(self, dataset, dataloader):
    self.dataset = dataset
    self.dataloader = dataloader

def get_dataset_and_dataloader(
  dataset=IMAGENET100,
  dataset_type='train',
  batch_size=2,
  transform=T.ToTensor(),
  datasets_path=DEFAULT_DATASETS_PATH
):
  parent_dataset_path = os.path.join(datasets_path, dataset)
  requested_dataset_path = os.path.join(parent_dataset_path, dataset_type)

  if not os.path.isdir(parent_dataset_path):
    raise Exception(f'Dataset {dataset} does not exist under {datasets_path} directory')
  if not os.path.isdir(requested_dataset_path):
    raise Exception(f'{dataset_type} set does not exist under {datasets_path}/{dataset} directory')

  dataset = datasets.ImageFolder(requested_dataset_path, transform=transform)

  kwargs = {"pin_memory": True, "num_workers": 1} if torch.cuda.is_available() else {}

  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

  return dataset, dataloader

