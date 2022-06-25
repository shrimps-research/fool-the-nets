import numpy as np
import torch
import time
from src.data.dataloaders import get_dataset_and_dataloader
from src.data.datasets import IMAGENET100
from torchvision import transforms as T
import os
from matplotlib import pyplot as plt
from pathlib import Path

from src.models.common import evaluate
from src.models.perceiver import get_perceiver_io, PERCEIVER_IO_LEARNED_POS_EMBEDDINGS
from src.models.vgg import get_vgg, VGG
from src.models.vit import get_vit, ViT
from src.models.swin import get_swin, SWIN
from src.models.xception import get_xception, XCEPTION
from src.attacks.white_box.transforms.fast_gradient import FastGradientTransform

DATASET = IMAGENET100
DATASET_DIR = 'train.X1'
CENTER_CROP_SIZE = 300
RESULTS_DIR = 'results'
RESULTS_FILE = 'results.csv'

SUPPORTED_MODEL_NAMES = [
  PERCEIVER_IO_LEARNED_POS_EMBEDDINGS,
  ViT,
  SWIN,
  VGG,
  XCEPTION
]

GET_MODEL_FUNCTION_BY_NAME = {
  PERCEIVER_IO_LEARNED_POS_EMBEDDINGS: get_perceiver_io,
  ViT: get_vit,
  SWIN: get_swin,
  VGG: get_vgg,
  XCEPTION: get_xception
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ToDevice(object):
  def __init__(self, device):
    self.device = device

  def __call__(self, sample):
    return sample.to(self.device)

class Detach(object):
  def __call__(self, sample):
    sample = sample.detach()
    sample.requires_grad = False
    return sample

def adversarial_attack(
  source_model_name,
  target_model_names,
  size,
  batch_size,
  methodToRun,
  kwargs,
):
  start_time = time.time()
  source_model = GET_MODEL_FUNCTION_BY_NAME[source_model_name](source_model_name)
  attack_transform = methodToRun(source_model.model, **kwargs)


  adversarial_train_set, adversarial_train_dataloader = get_dataset_and_dataloader(
    dataset=DATASET,
    dataset_type=DATASET_DIR,
    max_size=size,
    batch_size=batch_size,

    transform=T.Compose([
      T.CenterCrop(CENTER_CROP_SIZE),
      T.Resize(source_model.expected_image_size),
      T.ToTensor(),
      ToDevice(device),
      attack_transform,
      Detach()
    ]),
  )

  target_models = {name:get_target_model(source_model, source_model_name, name) for name in target_model_names}

  attack_results_per_model = evaluate(target_models, adversarial_train_dataloader)

  for target_model_name, attack_results in attack_results_per_model.items():
    dir_path = get_results_dir(attack_transform, source_model_name, target_model_name)
    print("Accuracy on attacked train set:", attack_results.accuracy)
    print("Average confidence on target class after attack:", np.mean(attack_results.confidence))

    if 'epsilon' in kwargs:
      run_name = f'{source_model_name}-{target_model_name}-{size}-{repr(attack_transform)}-epsilon_{kwargs["epsilon"]}'
    else:
      run_name = f'{source_model_name}-{target_model_name}-{size}-{repr(attack_transform)}-iter_{kwargs["max_iterations"]}'

    write_results(attack_results, dir_path, None, run_name, start_time)
    print(f'Time took for {source_model_name} -> {target_model_name}: {time.time() - start_time}')

  imagge_dir_path = get_images_dir(attack_transform, source_model_name)
  store_attacked_image(adversarial_train_dataloader, imagge_dir_path, image_name_from(kwargs, size))

def get_target_model(source_model, source_model_name, target_model_name):
  if source_model_name == target_model_name:
    target_model = source_model
  else:
    target_model = GET_MODEL_FUNCTION_BY_NAME[target_model_name](target_model_name)
  return target_model

def run_on_original_images(batch_size, size, source_model, target_model):
  train_set, train_dataloader = get_dataset_and_dataloader(
    dataset=DATASET,
    dataset_type=DATASET_DIR,
    max_size=size,
    batch_size=batch_size,
    transform=T.Compose([
      T.CenterCrop(CENTER_CROP_SIZE),
      T.Resize(source_model.expected_image_size),
      T.ToTensor(),
    ]),
  )
  results = evaluate(target_model, train_dataloader)
  print("Accuracy on train set:", results.accuracy)
  print("Average confidence on target class:", results.confidence)
  return results

def get_results_dir(attack_transform, source_model_name, target_model_name):
  Path(os.path.join(os.path.dirname(__file__), RESULTS_DIR)).mkdir(parents=True, exist_ok=True)
  dir_path = os.path.join(
    os.path.dirname(__file__),
    f'{RESULTS_DIR}/{repr(attack_transform)}-{source_model_name}-{target_model_name}'
  )
  Path(dir_path).mkdir(parents=True, exist_ok=True)
  return dir_path

def get_images_dir(attack_transform, source_model_name):
  Path(os.path.join(os.path.dirname(__file__), RESULTS_DIR)).mkdir(parents=True, exist_ok=True)
  dir_path = os.path.join(
    os.path.dirname(__file__),
    f'{RESULTS_DIR}/{repr(attack_transform)}-{source_model_name}'
  )
  Path(dir_path).mkdir(parents=True, exist_ok=True)
  return dir_path

def write_results(attack_results, dir_path, results, run_name, start_time):
  file_path = os.path.join(dir_path, RESULTS_FILE)
  if not Path(file_path).is_file():
    with open(file_path, 'a+') as f:
      f.write(
        f'accuracy, accuracy after attack, mean confidence, '
        f'mean confidence after attack, confidence std, confidence std after attack, run, time\n'
      )
  with open(file_path, 'a+') as f:
    if results is not None:
      f.write(
        f'{results.accuracy}, {attack_results.accuracy}, {results.confidence}, '
        f'{attack_results.confidence}, {results.confidence_std}, {attack_results.confidence_std}, {run_name}, {time.time() - start_time}\n'
      )
    else:
      f.write(
        f', {attack_results.accuracy}, , {attack_results.confidence}, , {attack_results.confidence_std}, {run_name}, {time.time() - start_time}\n'
      )

def store_attacked_image(adversarial_train_dataloader, dir_path, image_name):
  dataiter = iter(adversarial_train_dataloader)
  images, _ = dataiter.next()
  image = torch.tensor(images, requires_grad=False)[0].clone().detach().cpu().numpy()
  plt.imsave(os.path.join(dir_path, image_name), np.clip(np.transpose(image, (1, 2, 0)), 0, 1))

def image_name_from(kwargs, size):
  if 'epsilon' in kwargs:
    return f'epsilon_{kwargs["epsilon"]}_{size}.png'
  else:
    return f'max_iterations_{kwargs["max_iterations"]}_{size}.png'

if __name__ == '__main__':
  adversarial_attack('vgg', 'vgg', 2, 2, FastGradientTransform, kwargs={'epsilon': 0.03})