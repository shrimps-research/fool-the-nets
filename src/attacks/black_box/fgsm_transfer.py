import argparse

from src.attacks.adversarial_attack import adversarial_attack
from src.attacks.white_box.fgsm import SUPPORTED_MODEL_NAMES
from src.attacks.white_box.transforms.fast_gradient import FastGradientTransform
from src.models.perceiver import PERCEIVER_IO_LEARNED_POS_EMBEDDINGS
from src.models.vit import ViT

def parsed_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--source',
    help='Source model, used to create the adversarial attack',
    default=ViT,
    choices=SUPPORTED_MODEL_NAMES,
    type=str
  )
  parser.add_argument(
    '--target',
    help='Target model to be attacked',
    default=PERCEIVER_IO_LEARNED_POS_EMBEDDINGS,
    choices=SUPPORTED_MODEL_NAMES,
    type=str
  )
  parser.add_argument(
    '--epsilon',
    help='Epsilon parameter of FGSM',
    default=0.06, type=float
  )
  parser.add_argument(
    '--size', help='Number of images to attack', default=10, type=int
  )
  parser.add_argument(
    '--batch', help='Batch size', default=10, type=int
  )
  return parser.parse_args()


if __name__ == "__main__":
  args = parsed_args()
  adversarial_attack(
    args.source,
    args.target,
    args.size,
    args.batch,
    methodToRun=FastGradientTransform,
    kwargs={'epsilon': args.epsilon},
  )