import argparse

from src.attacks.black_box.transforms.ES_transform import ESTransform
from src.attacks.adversarial_attack import adversarial_attack
from src.attacks.black_box.es.Recombination import *
from src.attacks.black_box.es.Mutation import *
from src.attacks.black_box.es.Selection import *

def parsed_args():
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-eval', action='store', 
                        dest='evaluation', type=str,
                        default='crossentropy')
    parser.add_argument('-min', action='store_true', 
                        dest='minimize')
    parser.add_argument('-t', action='store_true', 
                        dest='targeted')
    parser.add_argument('-d', action='store',
                        dest='downsample', type=float,
                        default=None)
    parser.add_argument('-b', action='store', 
                        dest='budget', type=int,
                        default=10000)
    parser.add_argument('-model', action='store',
                        dest='model', type=str,
                        default='xception')
    parser.add_argument('-ps', action='store',
                        dest='parent_size', type=int,
                        default=12)
    parser.add_argument('-os', action='store', 
                        dest='offspring_size', type=int,
                        default=50)
    parser.add_argument('-r', action='store', 
                        dest='recombination', type=str,
                        default='global_discrete')
    parser.add_argument('-m', action='store', 
                        dest='mutation', type=str,
                        default='individual')
    parser.add_argument('-s', action='store', 
                        dest='selection', type=str,
                        default='comma_selection')
    parser.add_argument('-e', action='store', 
                        dest='epsilon', type=float,
                        default=0.05)
    parser.add_argument('-fp', action='store', 
                        dest='fallback_patience', type=int,
                        default=5)
    parser.add_argument('-sn', action='store',
                        dest='start_noise', type=str,
                        default=None)
    parser.add_argument('-v', action='store', 
                        dest='verbose', type=int,
                        default=2)
    parser.add_argument(
      '--size', help='Number of images to attack', default=2, type=int
    )
    parser.add_argument(
      '--batch', help='Batch size', default=2, type=int
    )
    args = parser.parse_args()
    
    if args.verbose:
        print("arguments passed:",args)
    
    return args


if __name__ == "__main__":
    recombinations = {  'intermediate': Intermediate(),
                        'discrete': Discrete(),
                        'global_discrete': GlobalDiscrete(),
                        None: None }

    mutations = {       'individual': IndividualSigma(),
                        'one_fifth': OneFifth(),
                        'one_fifth_alt': OneFifth(alt=True) }

    selections = {      'plus_selection': PlusSelection(),
                        'comma_selection': CommaSelection() }


    args = parsed_args()
    adversarial_attack(
        args.model,
        args.model,
        args.size,
        args.batch,
        methodToRun=ESTransform,
        kwargs={
            'minimize':args.minimize,
            'budget':args.budget,
            'parents_size':args.parent_size,
            'offspring_size':args.offspring_size,
            'recombination':recombinations[args.recombination],
            'mutation':mutations[args.mutation],
            'selection':selections[args.selection],
            'fallback_patience':args.fallback_patience,
            'verbose':args.verbose,
            'epsilon':args.epsilon,
            'downsample':args.downsample,
            'start_noise':args.start_noise
        },
    )