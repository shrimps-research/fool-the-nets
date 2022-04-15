import argparse
import numpy as np
from src.data.dataloaders import get_dataset_and_dataloader
from src.data.datasets import IMAGENET100
from torchvision import transforms as T

from src.models.common import evaluate
from src.models.perceiver import get_perceiver_io, PERCEIVER_IO_LEARNED_POS_EMBEDDINGS
from src.models.vit import get_vit, ViT
from src.attacks.white_box.transforms.carlini_wagner import CarliniWagnerL2

SUPPORTED_MODEL_NAMES = [
  PERCEIVER_IO_LEARNED_POS_EMBEDDINGS,
  ViT
]

GET_MODEL_FUNCTION_BY_NAME = {
  PERCEIVER_IO_LEARNED_POS_EMBEDDINGS: get_perceiver_io,
  ViT: get_vit
}

def adversarial_attack(
    source_model_name,
    target_model_name,
    size,
    batch_size,
    methodToRun, 
    kwargs,
):

    source_model = GET_MODEL_FUNCTION_BY_NAME[source_model_name](source_model_name)
    if source_model_name == target_model_name:
        target_model = source_model
    else:
        target_model = GET_MODEL_FUNCTION_BY_NAME[target_model_name](target_model_name)

    train_set, train_dataloader = get_dataset_and_dataloader(
        dataset=IMAGENET100,
        dataset_type='train.X1',
        max_size=size,
        batch_size=batch_size,
        transform=T.Compose([
        T.ToTensor(),
        T.Normalize(0, 1),
        T.CenterCrop(250),
        T.Resize(source_model.expected_image_size)
        ]),
    )

    # visualizer.show_image(train_dataloader)
    # visualizer.show_batch(train_dataloader)


    results = evaluate(
        target_model.model,
        train_dataloader,
        target_model.feature_extractor,
    )
    print("Accuracy on train set:", results.accuracy)
    print("Average confidence on target class:", np.mean(results.confidences, axis=0))

    adversarial_train_set, adversarial_train_dataloader = get_dataset_and_dataloader(
        dataset=IMAGENET100,
        dataset_type='train.X1',
        max_size=size,
        batch_size=batch_size,
        transform=T.Compose([
        T.ToTensor(),
        T.Normalize(0, 1),
        T.CenterCrop(250),
        T.Resize(source_model.expected_image_size),
        methodToRun(source_model.model, **kwargs)
        ]),
    )

    results = evaluate(
        target_model.model,
        adversarial_train_dataloader,
        target_model.feature_extractor,
    )
    print("Accuracy on train set attacked by FGSM method:", results.accuracy)
    print("Average confidence on target class after FGSM attack:", np.mean(results.confidences, axis=0))

