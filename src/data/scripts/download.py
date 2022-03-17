import argparse
import os
import wget
from zipfile import ZipFile

from src.data.datasets import KAGGLE_URLS, DATASET_NAMES, IMAGENET200, KAGGLE_DATASETS, WGET_DATASETS, WGET_URLS
from src.settings import DEFAULT_DATASETS_PATH


def download(dataset, destination_path):
  if dataset in KAGGLE_DATASETS:
    return kaggle_download(dataset, destination_path)
  elif dataset in WGET_DATASETS:
    return wget_download(dataset, destination_path)


def kaggle_download(dataset, destination_path):
  from kaggle.api.kaggle_api_extended import KaggleApi
  kaggle = KaggleApi()
  kaggle.authenticate()
  kaggle.dataset_download_files(KAGGLE_URLS[dataset], path=destination_path, unzip=True)
  return KAGGLE_URLS[dataset].split('/')[-1]


def wget_download(dataset, destination_path):
  wget.download(WGET_URLS[dataset], destination_path)
  downloaded_zip_name = WGET_URLS[dataset].split('/')[-1]
  with ZipFile(os.path.join(destination_path, downloaded_zip_name), 'r') as zipObj:
    zipObj.extractall(destination_path)

  return os.path.join(destination_path, downloaded_zip_name.split('.zip')[0])


def already_downloaded(dataset, destination_path):
  return os.path.isdir(os.path.join(destination_path, dataset))


def parsed_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', help='Dataset to download', default=IMAGENET200, choices=DATASET_NAMES, type=str)
  parser.add_argument('--path', help='Path to store the dataset', default=DEFAULT_DATASETS_PATH, type=str)
  return parser.parse_args()


if __name__ == "__main__":
  args = parsed_args()
  dataset = args.dataset
  destination_path = args.path

  if already_downloaded(dataset, destination_path):
    print(f'Dataset {dataset} is already downloaded on path: {os.path.join(destination_path, dataset)}.\n'
          f'Delete this directory if you want to download the dataset again.')
    exit(1)

  downloaded_dir_name = download(dataset, destination_path)
  os.rename(os.path.join(destination_path, downloaded_dir_name), os.path.join(args.path, args.dataset))