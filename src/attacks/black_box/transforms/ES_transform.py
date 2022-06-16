from torchvision import transforms
from src.attacks.black_box.es.Population import Population
from src.attacks.black_box.es.Evaluation import *
from src.attacks.black_box.es.Recombination import Recombination
from src.attacks.black_box.es.Mutation import Mutation
from src.attacks.black_box.es.Selection import Selection
from src.attacks.black_box.es.ES import ES

class ESTransform(torch.nn.Module, ES):

  def __init__(self, model, minimize: bool,
                budget: int, parents_size: int, offspring_size: int,
                recombination: Recombination, mutation: Mutation, selection: Selection,
                fallback_patience: int, verbose: int, epsilon=0.05, downsample=None,
                one_fifth=False, start_noise=None) -> None:
    super().__init__()
    self.model = model
    self.epsilon = epsilon
    self.minimize = minimize
    self.budget = budget
    self.parents_size = parents_size
    self.offspring_size = offspring_size
    self.recombination = recombination
    self.mutation = mutation
    self.selection = selection
    self.one_fifth = one_fifth
    self.fallback_patience = fallback_patience
    self.downsample = downsample
    self.start_noise = start_noise
    self.verbose=verbose
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
  def forward(self, img):
    img = img.unsqueeze(0)
    # img /= 255.
    normalized_img = self.normalize(img)
    logits = self.model(normalized_img)
    true_label = logits.argmax(-1).cpu().detach().numpy()

    self.evaluation = Crossentropy(self.model, true_label, minimize=self.minimize, targeted=False)
    self.parents = Population(img.numpy().transpose((0, 2, 3, 1)), self.parents_size, self.mutation, self.epsilon, self.downsample, self.start_noise)
    self.offspring = Population(img.numpy().transpose((0, 2, 3, 1)), self.offspring_size, self.mutation, self.epsilon, self.downsample, self.start_noise)

    parents, best_individual, _ = self.run()

    noise = parents.reshape_ind(best_individual)
    noise = parents.upsample_ind(noise)
    noise = torch.tensor(noise.transpose((2, 0, 1))).float().unsqueeze(0)
    noisy_image = (noise + img).clip(0, 1).squeeze()
    return noisy_image


  def __repr__(self) -> str:
    return f"{self.__class__.__name__}"


