import argparse
from src.attacks.white_box.fgsm import fgsm, SUPPORTED_MODEL_NAMES
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
    default=0.06, type=int
  )
  parser.add_argument(
    '--size', help='Number of images to attack', default=10, type=int
  )
  parser.add_argument(
    '--batch', help='Batch size', default=10, type=int
  )
  parser.add_argument(
    '--step', help='Epsilon parameter of PGD', default=0.005, type=int
  )
  parser.add_argument(
    '--iterations', help='Epsilon parameter of PGD', default=10, type=int
  )
  return parser.parse_args()


if __name__ == "__main__":
  args = parsed_args()
  fgsm(
    args.source,
    args.target,
    args.epsilon,
    args.size,
    args.batch
  )