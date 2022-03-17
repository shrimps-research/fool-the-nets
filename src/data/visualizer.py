from matplotlib import pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader


def imshow(img):
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


def show_batch(dataloader: DataLoader):
  dataiter = iter(dataloader)
  images, labels = dataiter.next()
  imshow(torchvision.utils.make_grid(images))


def show_image(dataloader: DataLoader):
  dataiter = iter(dataloader)
  images, labels = dataiter.next()
  random_num = np.random.randint(0, len(images) - 1)
  imshow(images[random_num])
  label = labels[random_num]
  print(f'Label: {label}, Shape: {images[random_num].shape}')