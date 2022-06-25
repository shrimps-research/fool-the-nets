import torch.optim
from tqdm import tqdm

from src.attacks.white_box.transforms.fast_gradient import FastGradientTransform
from src.data.dataloaders import get_dataset_and_dataloader
from src.data.datasets import IMAGENET100
from torchvision import transforms as T

from src.attacks.black_box.autoencoder.pretrained_autoencoder import PretrainedAutoencoder
from src.models.perceiver import get_perceiver_io, PERCEIVER_IO_LEARNED_POS_EMBEDDINGS
from src.models.vit import get_vit, ViT
from src.models.vgg import get_vgg, VGG
from src.models.swin import SWIN, get_swin


DATASET = IMAGENET100
DATASET_DIR = 'one-class'
TEST_SET_DIR = 'one-class-val'
CENTER_CROP_SIZE = 250
RESULTS_DIR = 'autoencoder-results'
RESULTS_FILE = 'autoencoder-results.txt'

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

def autoencoder_attack(
  target_model_name,
  size,
  batch_size,
):
  target_model = GET_MODEL_FUNCTION_BY_NAME[target_model_name](target_model_name)
  autoencoder = PretrainedAutoencoder()
  vit = get_vit().model
  fgsm = FastGradientTransform(vit, epsilon=0.05)


  train_set, train_dataloader = get_dataset_and_dataloader(
    dataset=DATASET,
    dataset_type=DATASET_DIR,
    max_size=1000,
    batch_size=8,
    transform=T.Compose([
      T.ToTensor(),
      T.CenterCrop(CENTER_CROP_SIZE),
      T.Resize(target_model.expected_image_size),
    ]),
  )

  batch_index = 0
  bce = torch.nn.BCELoss(reduction='mean')
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=0.1)

  for epoch in tqdm(range(200)):
    for batch in train_dataloader:
      batch_index += 1
      inputs = batch[0]
      labels = batch[1]
      inputs = torch.clip(inputs, min=0.0, max=1.0)
      noisy_images = T.Compose([fgsm])(inputs)
      targets = noisy_images - inputs
      # targets = noisy_images
      output = autoencoder.forward(inputs).clip(0,1)
      optimizer.zero_grad()
      noise_loss = bce(torch.flatten(output, start_dim=1), torch.flatten(targets, start_dim=1))
      noise_loss.backward()
      optimizer.step()

if __name__ == "__main__":
  autoencoder_attack('swin', 128, 16)
