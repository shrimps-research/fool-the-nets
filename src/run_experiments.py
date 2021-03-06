from src.attacks.white_box.transforms.fast_gradient import FastGradientTransform
from src.attacks.white_box.transforms.projected_gradient import ProjectedGradientTransform
from src.attacks.white_box.transforms.carlini_wagner import CarliniWagnerL2
from src.attacks.black_box.transforms.random_noise import GaussianNoiseTransform
from src.data.datasets import IMAGENET100
from src.attacks.black_box.fgsm_transfer import adversarial_attack
import concurrent.futures
import time
import multiprocessing

DATASET = IMAGENET100
DATASET_DIR = 'train.X1'
SAMPLES_SIZE = 5000
BATCH_SIZE = 2
EPSILON = 0.025

run_configs = [
  ('vit', ['xception', 'swin', 'vgg', 'vit', 'perceiver-io']),
  ('xception', ['xception', 'swin', 'vgg', 'vit', 'perceiver-io']),
  ('vgg', ['xception', 'swin', 'vgg', 'vit', 'perceiver-io']),
  ('swin', ['xception', 'swin', 'vgg', 'vit', 'perceiver-io']),
  ('perceiver-io', ['xception', 'swin', 'vgg', 'vit', 'perceiver-io']),
]

def attack(source, target):
  adversarial_attack(
    source,
    target,
    SAMPLES_SIZE,
    BATCH_SIZE,
    methodToRun=ProjectedGradientTransform,
    kwargs={'epsilon': EPSILON, 'step_size': EPSILON/10, 'iterations': 50},
    # kwargs={'epsilon': 0.05},
  )


if __name__ == '__main__':
  try:
    multiprocessing.set_start_method('spawn')
  except RuntimeError:
    pass

  start_time = time.time()
  with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    runs = [executor.submit(attack, *run_config) for run_config in run_configs]
    for run in concurrent.futures.as_completed(runs):
      try:
        run.result()
      except Exception as ex:
        print('Exception was thrown: ', ex)

  end_time = time.time()
  print(f'Time: {end_time - start_time}')
