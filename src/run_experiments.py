from src.attacks.white_box.transforms.fast_gradient import FastGradientTransform
from src.data.datasets import IMAGENET100
from src.attacks.black_box.fgsm_transfer import adversarial_attack
import concurrent.futures
import itertools
import time

DATASET = IMAGENET100
DATASET_DIR = 'train.X1'
SAMPLES_SIZE = 2048
BATCH_SIZE = 16
EPSILON = 0.05

models = [
  'vit',
  'perceiver-io',
  'swin',
  'vgg',
  'xception',
]

run_configs = list(itertools.permutations(models, 2))

def attack(source, target):
  adversarial_attack(
    source,
    target,
    SAMPLES_SIZE,
    BATCH_SIZE,
    methodToRun=FastGradientTransform,
    kwargs={'epsilon': EPSILON},
  )


if __name__ == '__main__':
  start_time = time.time()
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    runs = [executor.submit(attack, *run_config) for run_config in run_configs]
    for run in concurrent.futures.as_completed(runs):
      try:
        run.result()
      except Exception as ex:
        print('Exception was thrown: ', ex)
  end_time = time.time()
  print(f'Time: {end_time - start_time}')
