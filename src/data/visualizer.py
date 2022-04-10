import torch
from matplotlib import pyplot as plt
import numpy as np
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader


def imshow(img):
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


def show_batch(dataloader: DataLoader):
  dataiter = iter(dataloader)
  images, labels = dataiter.next()
  imshow(torchvision.utils.make_grid(images))


def show_image(dataloader: DataLoader, offset=0):
  dataiter = iter(dataloader)
  for _ in range(offset+1):
    image, label = dataiter.next()
  image = Variable(image, requires_grad=False)
  random_num = np.random.randint(0, len(image) - 1)
  imshow(image[random_num])
  label = label[random_num]
  print(f'Label: {label}, Shape: {image[random_num].shape}')