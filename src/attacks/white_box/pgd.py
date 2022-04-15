import argparse
import numpy as np
from torchvision import transforms as T

from src.attacks.white_box.transforms.projected_gradient import ProjectedGradientTransform
from src.data.dataloaders import get_dataset_and_dataloader
from src.data import visualizer
from src.data.datasets import IMAGENET100
from src.models.common import evaluate
from src.models.perceiver import get_perceiver_io, PERCEIVER_IO_LEARNED_POS_EMBEDDINGS
from src.models.vit import get_vit, ViT
from src.attacks.adversarial_attack import adversarial_attack

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
    '--epsilon', help='Epsilon parameter of PGD', default=0.03, type=float
  )
  parser.add_argument(
    '--step', help='Epsilon parameter of PGD', default=0.005, type=float
  )
  parser.add_argument(
    '--iterations', help='Epsilon parameter of PGD', default=10, type=int
  )
  parser.add_argument(
    '--size', help='Number of images to attack', default=10, type=int
  )
  parser.add_argument(
    '--batch', help='Batch size', default=5, type=int
  )
  return parser.parse_args()


if __name__ == "__main__":
  args = parsed_args()
  adversarial_attack(
    args.model,
    args.model,
    args.size,
    args.batch,
    methodToRun=ProjectedGradientTransform,
    kwargs={
            'epsilon': args.epsilon,
            'step_size': args.step,
            'iterations': args.iterations
          },
  )
