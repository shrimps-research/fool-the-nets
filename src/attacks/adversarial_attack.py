import numpy as np
from src.data.dataloaders import get_dataset_and_dataloader
from src.data.datasets import IMAGENET100
from torchvision import transforms as T
from torch.autograd import Variable
import os
from matplotlib import pyplot as plt
from pathlib import Path

from src.models.common import evaluate
from src.models.perceiver import get_perceiver_io, PERCEIVER_IO_LEARNED_POS_EMBEDDINGS
from src.models.vit import get_vit, ViT

DATASET = IMAGENET100
DATASET_DIR = 'train.X1'
CENTER_CROP_SIZE = 300
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

def adversarial_attack(
    source_model_name,
    target_model_name,
    size,
    batch_size,
    methodToRun,
    kwargs,
    include_original_accuracy=False
):

    source_model = GET_MODEL_FUNCTION_BY_NAME[source_model_name](source_model_name)
    target_model = get_target_model(source_model, source_model_name, target_model_name)
    attack_transform = methodToRun(source_model.model, **kwargs)


    if include_original_accuracy:
        results = run_on_original_images(batch_size, size, source_model, target_model)
    else:
        results = None


    adversarial_train_set, adversarial_train_dataloader = get_dataset_and_dataloader(
        dataset=DATASET,
        dataset_type=DATASET_DIR,
        max_size=size,
        batch_size=batch_size,
        transform=T.Compose([
            T.ToTensor(),
            T.CenterCrop(CENTER_CROP_SIZE),
            T.Resize(source_model.expected_image_size),
            attack_transform,
        ]),
    )
    attack_results = evaluate(target_model.model, adversarial_train_dataloader)
    print("Accuracy on attacked train set:", attack_results.accuracy)
    print("Average confidence on target class after attack:", np.mean(attack_results.confidence))


    dir_path = get_results_dir(attack_transform, source_model_name, target_model_name)
    run_name = f'{source_model_name}-{target_model_name}-{size}-{repr(attack_transform)}-{kwargs}'
    write_results(attack_results, dir_path, results, run_name)
    store_attacked_image(adversarial_train_dataloader, dir_path, image_name_from(kwargs, size))


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
            T.ToTensor(),
            T.CenterCrop(CENTER_CROP_SIZE),
            T.Resize(source_model.expected_image_size)
        ]),
    )
    results = evaluate(target_model.model, train_dataloader)
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


def write_results(attack_results, dir_path, results, run_name):
    file_path = os.path.join(dir_path, RESULTS_FILE)
    if not Path(file_path).is_file():
        with open(file_path, 'a+') as f:
            f.write(
                f'accuracy, accuracy after attack, mean confidence, '
                f'mean confidence after attack, confidence std, confidence std after attack, run\n'
            )
    with open(file_path, 'a+') as f:
        if results is not None:
            f.write(
                f'{results.accuracy}, {attack_results.accuracy}, {results.confidence}, '
                f'{attack_results.confidence}, {results.confidence_std}, {attack_results.confidence_std}, {run_name}\n'
            )
        else:
            f.write(
                f', {attack_results.accuracy}, , {attack_results.confidence}, , {attack_results.confidence_std}, {run_name}\n'
            )


def store_attacked_image(adversarial_train_dataloader, dir_path, image_name):
    dataiter = iter(adversarial_train_dataloader)
    images, _ = dataiter.next()
    image = Variable(images, requires_grad=False)[0].numpy()
    plt.imsave(os.path.join(dir_path, image_name), np.clip(np.transpose(image, (1, 2, 0)), 0, 1))


def image_name_from(attack_args, size):
    return f'{attack_args}_{size}.png'
