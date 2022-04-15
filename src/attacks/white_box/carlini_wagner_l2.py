import argparse
import numpy as np
from src.data.dataloaders import get_dataset_and_dataloader
from src.data.datasets import IMAGENET100
from torchvision import transforms as T

from src.models.common import evaluate
from src.models.perceiver import get_perceiver_io, PERCEIVER_IO_LEARNED_POS_EMBEDDINGS
from src.models.vit import get_vit, ViT
from src.attacks.adversarial_attack import adversarial_attack
from src.attacks.white_box.transforms.carlini_wagner import CarliniWagnerL2
import os

DEFAULT_CONFIG_PATH = os.getenv("DEFAULT_CONFIG_PATH")



SUPPORTED_MODEL_NAMES = [
  PERCEIVER_IO_LEARNED_POS_EMBEDDINGS,
  ViT
]


def parsed_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--model',
    help='Model to attack',
    default=ViT,
    choices=SUPPORTED_MODEL_NAMES,
    type=str
  )

  parser.add_argument(
    '--size', help='Number of images to attack', default=10, type=int
  )
  parser.add_argument(
    '--batch', help='Batch size', default=5, type=int
  )
  parser.add_argument(
    '--n_classes', help='Number of classes', default=1_000, type=int
  )
  parser.add_argument(
    '--max_iterations', help='Number of classes', default=1_000, type=int
  )
  parser.add_argument(
    '--binary_search_steps', help='Number of times to perform binary search', default=5, type=int
  )
  return parser.parse_args()

if __name__ == "__main__":
  args = parsed_args()
  adversarial_attack(
    args.model,
    args.model,
    args.size,
    args.batch,
    methodToRun=CarliniWagnerL2,
    kwargs={
            'n_classes': args.n_classes,
            'max_iterations': args.max_iterations,
            'binary_search_steps': args.binary_search_steps
          },
  )
