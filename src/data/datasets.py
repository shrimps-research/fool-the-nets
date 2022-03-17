IMAGENET100 = 'imagenet100'
IMAGENET200 = 'tiny-imagenet200'  # 100,000 images of 200 classes (500 for each class) downsized to 64×64 colored images

KAGGLE_URLS = {
  # IMAGENET10: 'c/diabetic-retinopathy-detection'
  IMAGENET100: 'ambityga/imagenet100'
}
KAGGLE_DATASETS = list(KAGGLE_URLS.keys())

WGET_URLS = {
  IMAGENET200: 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
}
WGET_DATASETS = list(WGET_URLS.keys())

DATASET_NAMES = KAGGLE_DATASETS + WGET_DATASETS