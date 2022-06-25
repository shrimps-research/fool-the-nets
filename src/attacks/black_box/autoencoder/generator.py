from pathlib import Path

import numpy as np
import torch.optim
from tqdm import tqdm
from datasets import load_metric
from matplotlib import pyplot as plt
import os

from src.attacks.black_box.autoencoder.generator_model import NoiseGenerator
from src.data.dataloaders import get_dataset_and_dataloader
from src.data.datasets import IMAGENET100
from torchvision import transforms as T
import torch.nn.functional as F

from src.models.perceiver import get_perceiver_io, PERCEIVER_IO_LEARNED_POS_EMBEDDINGS
from src.models.vit import get_vit, ViT
from src.models.vgg import get_vgg, VGG
from src.models.swin import SWIN, get_swin

DATASET = IMAGENET100
DATASET_DIR = 'one-class'
TEST_SET_DIR = 'one-class-val'
CENTER_CROP_SIZE = 250
RESULTS_DIR = 'results'
RESULTS_FILE = 'results.txt'

SUPPORTED_MODEL_NAMES = [
  PERCEIVER_IO_LEARNED_POS_EMBEDDINGS,
  ViT,
  VGG
]

GET_MODEL_FUNCTION_BY_NAME = {
  PERCEIVER_IO_LEARNED_POS_EMBEDDINGS: get_perceiver_io,
  ViT: get_vit,
  VGG: get_vgg,
  SWIN: get_swin
}

def generator_attack(
  source_model_name,
  target_model_name,
):
  source_model = GET_MODEL_FUNCTION_BY_NAME[source_model_name](source_model_name)
  generator = NoiseGenerator(source_model.expected_image_size)
  optimizer = torch.optim.Adam(generator.parameters())

  train_set, train_dataloader = get_dataset_and_dataloader(
    dataset=DATASET,
    dataset_type=DATASET_DIR,
    max_size=5000,
    batch_size=8,
    transform=T.Compose([
      T.RandomResizedCrop(224),
      T.RandomHorizontalFlip(),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
  )

  dir_path = get_results_dir()
  batch_index = 0
  bce = torch.nn.BCELoss(reduction='mean')

  for epoch in tqdm(range(200)):
    for batch in train_dataloader:

      batch_index += 1
      inputs = batch[0]
      labels = batch[1]
      inputs = torch.clip(inputs, min=0.0, max=1.0)

      # if epoch==10:
      #   generator.enable_noise()
      # adversarial_images = generator(inputs)

      # noises = generator(inputs)
      adversarial_images = generator(inputs)
      adversarial_images = torch.clip(adversarial_images, min=0.0, max=1.0)

      logits = source_model.model(adversarial_images)

      probs = F.softmax(logits, dim=1)
      batch_confidences = probs.gather(1, labels.unsqueeze(1))
      optimizer.zero_grad()

      # noise_loss = bce(inputs, adversarial_images)
      noise_loss = bce(torch.flatten(inputs, start_dim=1), torch.flatten(adversarial_images, start_dim=1))
      # noise_loss = 0
      batch_confidence_loss = torch.mean(batch_confidences)

      # loss = noise_loss + batch_confidence_loss
      if epoch < 20:
        loss = noise_loss
      else:
        loss = noise_loss + batch_confidence_loss

      loss.backward()

      optimizer.step()

      if batch_index % 10 == 0:
        mean_noise = torch.mean(torch.abs(adversarial_images - inputs)).cpu().detach().numpy().item()
        max_noise = torch.max(torch.abs(adversarial_images - inputs)).cpu().detach().numpy().item()

        image = adversarial_images[0].cpu().detach().numpy()
        plt.imsave(os.path.join(dir_path, f'attacked-e{epoch}-b{batch_index}.png'),
                   np.clip(np.transpose(image, (1, 2, 0)), 0, 1))
        predictions = logits.argmax(-1).cpu().detach().numpy()
        references = labels.numpy()
        accuracy = load_metric("accuracy")
        accuracy.add_batch(predictions=predictions, references=references)
        print(
          f"Batch accuracy: {accuracy.compute()['accuracy']}, "
          f"Loss: {loss.cpu().detach().numpy().item()}, Noise mean: {mean_noise}, "
          f"Noise max: {max_noise}"
        )

  test_set, test_dataloader = get_dataset_and_dataloader(
    dataset=DATASET,
    dataset_type=TEST_SET_DIR,
    max_size=16,
    batch_size=4,
    transform=T.Compose([
      T.RandomResizedCrop(224),
      T.RandomHorizontalFlip(),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
  )

  test_accuracy = load_metric("accuracy")

  for batch in tqdm(test_dataloader):
    inputs = batch[0]
    labels = batch[1]

    adversarial_images = inputs
    adversarial_images = torch.clip(adversarial_images, min=0.0, max=1.0)
    logits = source_model.model(adversarial_images)

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
    f'{RESULTS_DIR}/one-class-noisy-img-0,08_generator-cnn_swin-swin'
  )
  Path(dir_path).mkdir(parents=True, exist_ok=True)
  return dir_path

if __name__ == "__main__":
  generator_attack('swin', 'swin', 128, 16)
