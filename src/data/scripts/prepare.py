import argparse
import os

from src.data.datasets import DATASET_NAMES, IMAGENET200
from src.settings import DEFAULT_DATASET_PATH


def prepare_imagenet200(dataset, destination_path):
  print(f'Preparing structure for {dataset} structure')
  dataset_dir = os.path.join(destination_path, dataset)
  val_dir = os.path.join(dataset_dir, 'val')
  with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as fp:
    data = fp.readlines()

    val_img_dict = {}
    for line in data:
      words = line.split('\t')
      val_img_dict[words[0]] = words[1]

  val_img_dir = os.path.join(val_dir, 'images')
  for img, folder in val_img_dict.items():
    newpath = (os.path.join(val_img_dir, folder))
    if not os.path.exists(newpath):
      os.makedirs(newpath)
    if os.path.exists(os.path.join(val_img_dir, img)):
      os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))


def dataset_does_not_exist():
  os.path.isdir(os.path.join(destination_path, dataset))


def parsed_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', help='Dataset to prepare', default=IMAGENET200, choices=DATASET_NAMES, type=str)
  parser.add_argument('--path', help='Path where dataset is stored', default=DEFAULT_DATASET_PATH, type=str)
  return parser.parse_args()


HANDLER_PER_DATASET = {
  IMAGENET200: prepare_imagenet200
}

if __name__ == "__main__":
  args = parsed_args()
  dataset = args.dataset
  destination_path = args.path

  if dataset_does_not_exist():
    print(f'Dataset {dataset} does not exist in path: {os.path.join(destination_path, dataset)}.')
    exit(1)

  if dataset not in HANDLER_PER_DATASET:
    print(f'No handler exists for dataset {dataset}.')
    exit(2)

  HANDLER_PER_DATASET[dataset](dataset, destination_path)
