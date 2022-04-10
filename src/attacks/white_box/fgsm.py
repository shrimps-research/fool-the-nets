import argparse
import numpy as np
from src.data.dataloaders import get_dataset_and_dataloader
from src.data.datasets import IMAGENET100
from torchvision import transforms as T

from src.models.common import evaluate
from src.models.perceiver import get_perceiver_io, PERCEIVER_IO_LEARNED_POS_EMBEDDINGS
from src.models.vit import get_vit, ViT
from src.attacks.white_box.transforms.fast_gradient import FastGradientTransform

SUPPORTED_MODEL_NAMES = [
  PERCEIVER_IO_LEARNED_POS_EMBEDDINGS,
  ViT
]

GET_MODEL_FUNCTION_BY_NAME = {
  PERCEIVER_IO_LEARNED_POS_EMBEDDINGS: get_perceiver_io,
  ViT: get_vit
}

def fgsm(
  source_model_name,
  target_model_name,
  epsilon,
  size,
  batch_size
):

  source_model = GET_MODEL_FUNCTION_BY_NAME[source_model_name](source_model_name)
  if source_model_name == target_model_name:
    target_model = source_model
  else:
    target_model = GET_MODEL_FUNCTION_BY_NAME[target_model_name](target_model_name)

  train_set, train_dataloader = get_dataset_and_dataloader(
    dataset=IMAGENET100,
    dataset_type='train.X1',
    max_size=size,
    batch_size=batch_size,
    transform=T.Compose([
      T.ToTensor(),
      T.Normalize(0, 1),
      T.CenterCrop(250),
      T.Resize(source_model.expected_image_size)
    ]),
  )

  # visualizer.show_image(train_dataloader)
  # visualizer.show_batch(train_dataloader)


  results = evaluate(
    target_model.model,
    train_dataloader,
    target_model.feature_extractor,
  )
  print("Accuracy on train set:", results.accuracy)
  print("Average confidence on target class:", np.mean(results.confidences, axis=0))

  adversarial_train_set, adversarial_train_dataloader = get_dataset_and_dataloader(
    dataset=IMAGENET100,
    dataset_type='train.X1',
    max_size=size,
    batch_size=batch_size,
    transform=T.Compose([
      T.ToTensor(),
      T.Normalize(0, 1),
      T.CenterCrop(250),
      T.Resize(source_model.expected_image_size),
      FastGradientTransform(source_model.model, epsilon=epsilon)
    ]),
  )

  # visualizer.show_image(adversarial_train_dataloader)
  # visualizer.show_batch(adversarial_train_dataloader)

  results = evaluate(
    target_model.model,
    adversarial_train_dataloader,
    target_model.feature_extractor,
  )
  print("Accuracy on train set attacked by FGSM method:", results.accuracy)
  print("Average confidence on target class after FGSM attack:", np.mean(results.confidences, axis=0))


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
    '--epsilon', help='Epsilon parameter of FGSM', default=0.03, type=float
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
  fgsm(
    args.model,
    args.model,
    args.epsilon,
    args.size,
    args.batch
  )
