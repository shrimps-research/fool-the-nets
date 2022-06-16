from src.attacks.black_box.es.Population import Population
from src.attacks.black_box.es.Evaluation import Evaluation
from src.attacks.black_box.es.Recombination import Recombination
from src.attacks.black_box.es.Mutation import Mutation
from src.attacks.black_box.es.Selection import Selection
import numpy as np


class ES:
    """ Main Evolutionary Strategy class
    """
    def __init__(self, input_: np.ndarray, evaluation: Evaluation, minimize: bool,
                budget: int, parents_size: int, offspring_size: int,
                recombination: Recombination, mutation: Mutation, selection: Selection,
                fallback_patience: int, verbose: int, epsilon=0.05, downsample=None,
                one_fifth=False, start_noise=None) -> None:
        self.evaluation = evaluation
        self.minimize = minimize
        self.budget = budget
        self.parents_size = parents_size
        self.offspring_size = offspring_size
        self.recombination = recombination
        self.mutation = mutation
        self.selection = selection
        self.one_fifth = one_fifth
        self.fallback_patience = fallback_patience
        self.verbose=verbose
        # one_sigma = True if mutation.__class__.__name__ == "OneSigma" else False
        self.parents = Population(input_, self.parents_size, mutation, epsilon, downsample, start_noise)
        self.offspring = Population(input_, self.offspring_size, mutation, epsilon, downsample, start_noise)

    def run(self):
        """ Main function to run the Evolutionary Strategy
        """
        # initialize budget and best evaluation (as worst possible)
        curr_budget = 0
        curr_patience = 0
        best_eval = self.evaluation.worst_eval()

        # initialize (generation-wise) success probability params
        # success means finding a new best individual in a given gen. of offspring
        # gen_tot=num. of offspring gen., gen_succ=num. of successfull gen.
        gen_tot = 0
        gen_succ = 0

        # initial parents evaluation step
        self.parents.evaluate(self.evaluation.evaluate)
        best_eval, best_index = self.parents.best_fitness(self.minimize)
        best_indiv = self.parents.individuals[best_index]
        curr_budget += self.parents_size

        while curr_budget < self.budget:
            gen_tot += 1            
            
            # recombination: creates new offspring
            if self.recombination is not None and self.parents_size > 1:
                self.recombination(self.parents, self.offspring)
            
            # mutation: mutate individuals (offspring)
            self.mutation(self.offspring, gen_succ, gen_tot)
            
            # evaluate offspring population
            self.offspring.evaluate(self.evaluation.evaluate)
            curr_budget += self.offspring_size
            curr_patience += self.offspring_size  # TODO patience
            
            # next generation parents selection
            self.selection(self.parents, self.offspring, self.minimize)
            
            # update the best individual in case of success
            curr_best_eval = self.parents.fitnesses[0]
            
            success = False
            if self.minimize:
                if curr_best_eval < best_eval:
                    success = True
            else:
                if curr_best_eval > best_eval:
                    success = True
            if success:
                gen_succ += 1
                best_indiv = self.parents.individuals[0]
                best_eval = curr_best_eval
                curr_patience = 0  # reset patience since we found a new best
                if self.verbose > 1:
                    if self.evaluation.__class__.__name__ == "BlindEvaluation":  # TODO fix this print (pred is wrong)
                        print(f"[{curr_budget}/{self.budget}] New best eval: {round(best_eval, 4)}" + \
                        f" | Pred: {np.exp(best_eval - 0.01*np.log(np.sum(self.parents.individuals[0])/self.parents.individuals[0].max()))}" + \
                        f" | P_succ: {round(gen_succ/gen_tot, 2)}")
                    else:
                        print(f"[{curr_budget}/{self.budget}] New best eval: {round(best_eval, 4)}" + \
                        f" | Pred: {round(np.abs(np.exp(best_eval)),2)} | P_succ: {round(gen_succ/gen_tot, 2)}")
            else:
                curr_patience += 1
                if self.verbose > 2:
                    print(f"Gen {gen_tot}, no best found")

            # reset sigmas if patience expired
            if self.fallback_patience != None and curr_patience >= self.fallback_patience:
                self.parents.init_sigmas()

        return self.parents, best_indiv, best_eval

    def epsilon_annealing(self, curr_budget, annealing_step):
        """ Testing implementation of the epsilon annealing
        """
        if curr_budget/self.budget > 0.5:
            if annealing_step < 3:
                annealing_step = 3
                self.parents.epsilon = 0.02
                self.offspring.epsilon = self.parents.epsilon
                best_eval = self.evaluation.worst_eval()
                self.parents.evaluate(self.evaluation.evaluate)
                return  best_eval
        elif curr_budget/self.budget > 0.25:
            if annealing_step < 2:
                annealing_step = 2
                self.parents.epsilon = 0.05
                self.offspring.epsilon = self.parents.epsilon
                best_eval = self.evaluation.worst_eval()
                self.parents.evaluate(self.evaluation.evaluate)
                return  best_eval
        elif curr_budget/self.budget > 0.1:
            if annealing_step < 1:
                annealing_step = 1
                self.parents.epsilon = 0.1
                self.offspring.epsilon = self.parents.epsilon
                best_eval = self.evaluation.worst_eval()
                self.parents.evaluate(self.evaluation.evaluate)
                return  best_eval
        elif annealing_step < 0:
            annealing_step = 0
            self.parents.epsilon = 0.2
            self.offspring.epsilon = self.parents.epsilon