import torch
from src.data.dataloaders import get_dataset_and_dataloader
from src.data import visualizer
from src.data.datasets import IMAGENET100
from src.models.perceiver import PRETRAINED_MODELS, VISION_LEARNED_POS_EMBEDDINGS, PretrainedModel
from src.models.perceiver import evaluate
from transformers import PerceiverForImageClassificationLearned
from torchvision import transforms as T

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  pretrained_model: PretrainedModel = PRETRAINED_MODELS[VISION_LEARNED_POS_EMBEDDINGS]

  train_set, train_dataloader = get_dataset_and_dataloader(
    dataset=IMAGENET100,
    batch_size=2,
    transform=T.Compose([T.ToTensor(), T.CenterCrop(350), T.Resize(pretrained_model.expected_image_size)]),
  )
  visualizer.show_batch(train_dataloader)

  idx_to_class = {idx: label for idx, label in enumerate(train_set.classes)}
  class_to_idx = train_set.class_to_idx

  model = PerceiverForImageClassificationLearned.from_pretrained(
    pretrained_model.weights,
    num_labels=200,
    id2label=class_to_idx,
    label2id=idx_to_class,
    ignore_mismatched_sizes=True
  )
  model.to(device)

  accuracy = evaluate(model, pretrained_model.feature_extractor, train_dataloader)
  print("Accuracy on test set:", accuracy)



if __name__ == "__main__":
  main()
