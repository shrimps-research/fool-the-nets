from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import torch
import os

from src.data.datasets import IMAGENET200
from src.settings import DEFAULT_DATASETS_PATH

def generate_dataloader(
  dataset=IMAGENET200,
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

  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    **kwargs
  )

