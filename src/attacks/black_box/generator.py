from pathlib import Path

import numpy as np
import torch.optim
from tqdm import tqdm
from datasets import load_metric
from matplotlib import pyplot as plt
import os

from src.data.dataloaders import get_dataset_and_dataloader
from src.data.datasets import IMAGENET100
from torchvision import transforms as T
import torch.nn.functional as F

from src.models.generator import NoiseGenerator
from src.models.perceiver import get_perceiver_io, PERCEIVER_IO_LEARNED_POS_EMBEDDINGS
from src.models.vit import get_vit, ViT

DATASET = IMAGENET100
DATASET_DIR = 'train.X1'
TEST_SET_DIR = 'train.X2'
CENTER_CROP_SIZE = 250
RESULTS_DIR = 'results'
RESULTS_FILE = 'results.txt'

SUPPORTED_MODEL_NAMES = [
  PERCEIVER_IO_LEARNED_POS_EMBEDDINGS,
  ViT
]

GET_MODEL_FUNCTION_BY_NAME = {
  PERCEIVER_IO_LEARNED_POS_EMBEDDINGS: get_perceiver_io,
  ViT: get_vit
}

def generator_attack(
  source_model_name,
  target_model_name,
  size,
  batch_size,
  include_original_accuracy=False
):
  source_model = GET_MODEL_FUNCTION_BY_NAME[source_model_name](source_model_name)
  target_model = get_target_model(source_model, source_model_name, target_model_name)
  generator = NoiseGenerator(source_model.expected_image_size)
  optimizer = torch.optim.Adam(generator.model.parameters())

  train_set, train_dataloader = get_dataset_and_dataloader(
    dataset=DATASET,
    dataset_type=DATASET_DIR,
    max_size=1024,
    batch_size=16,
    transform=T.Compose([
      T.ToTensor(),
      T.CenterCrop(CENTER_CROP_SIZE),
      T.Resize(source_model.expected_image_size),
    ]),
  )

  dir_path = get_results_dir()
  batch_index = 0
  bce = torch.nn.BCELoss(reduction='mean')
  for epoch in tqdm(range(100)):
    for batch in train_dataloader:
      batch_index += 1
      inputs = batch[0]
      labels = batch[1]

      # adversarial_images = generator(inputs)
      noises = generator(inputs)
      adversarial_images = torch.clamp(inputs + noises, min=0.0, max=1.0)

      logits = source_model.model(adversarial_images)

      probs = F.softmax(logits, dim=1)
      batch_confidences = probs.gather(1, labels.unsqueeze(1))
      optimizer.zero_grad()
      noise_loss = bce(adversarial_images, inputs)
      # noise_loss = bce(torch.flatten(inputs, start_dim=1), torch.flatten(adversarial_images, start_dim=1))

      loss = torch.mean(batch_confidences)
      # loss.retain_grad()
      # loss = noise_loss
      # loss.retain_grad()
      # loss = torch.mean(batch_confidences)
      loss.backward()
      optimizer.step()

      if batch_index % 10 == 0:
        image = adversarial_images[0].cpu().detach().numpy()
        plt.imsave(os.path.join(dir_path, f'attacked-e{epoch}-b{batch_index}.png'),
                   np.clip(np.transpose(image, (1, 2, 0)), 0, 1))
        predictions = logits.argmax(-1).cpu().detach().numpy()
        references = labels.numpy()
        accuracy = load_metric("accuracy")
        accuracy.add_batch(predictions=predictions, references=references)
        print(
          f"Batch accuracy: {accuracy.compute()['accuracy']}, Loss: {loss.cpu().detach().numpy().item()}, Noise mean: {noises.abs().mean().cpu().detach().numpy().item()}")
        # print(f"Batch accuracy: {accuracy.compute()['accuracy']}, Loss: {loss.cpu().detach().numpy().item()} = {noise_loss.cpu().detach().numpy().item()} + {5 * torch.mean(batch_confidences).cpu().detach().numpy().item()}")

  test_set, test_dataloader = get_dataset_and_dataloader(
    dataset=DATASET,
    dataset_type=TEST_SET_DIR,
    max_size=16,
    batch_size=4,
    transform=T.Compose([
      T.ToTensor(),
      T.CenterCrop(CENTER_CROP_SIZE),
      T.Resize(source_model.expected_image_size),
    ]),
  )

  test_accuracy = load_metric("accuracy")
  # confidences = np.array([])

  for batch in tqdm(test_dataloader):
    inputs = batch[0]
    labels = batch[1]

    noises = generator(inputs)
    adversarial_images = inputs + noises
    logits = source_model.model(adversarial_images)

    # probs = F.softmax(logits, dim=1)
    # batch_confidences = probs.gather(1, labels.unsqueeze(1)).cpu().detach().numpy()
    # confidences = np.concatenate((confidences, np.squeeze(batch_confidences)))

    predictions = logits.argmax(-1).cpu().detach().numpy()
    references = labels.numpy()
    test_accuracy.add_batch(predictions=predictions, references=references)

  print(f"Test accuracy: {test_accuracy.compute()['accuracy']}")
  # print(f"Test confidence: {np.mean(confidences)}")

def get_target_model(source_model, source_model_name, target_model_name):
  if source_model_name == target_model_name:
    target_model = source_model
  else:
    target_model = GET_MODEL_FUNCTION_BY_NAME[target_model_name](target_model_name)
  return target_model

def get_results_dir():
  dir_path = os.path.join(
    os.path.dirname(__file__),
    f'{RESULTS_DIR}/e0.05_generator-cnn_vit-vit'
  )
  Path(dir_path).mkdir(parents=True, exist_ok=True)
  return dir_path

if __name__ == "__main__":
  generator_attack('vit', 'vit', 16, 4)
