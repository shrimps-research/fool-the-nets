# Carlini-Wagner Runs
MODELS=$(yq '.configs.models[]' 'configurations/carlini_wagner.yaml')
SIZES=$(yq '.configs.sizes[]' 'configurations/carlini_wagner.yaml')
BATCHES=$(yq '.configs.batches[]' 'configurations/carlini_wagner.yaml')
N_CLASSES=$(yq '.configs.n_classes[]' 'configurations/carlini_wagner.yaml')
MAX_ITERATIONS=$(yq '.configs.max_iterations[]' 'configurations/carlini_wagner.yaml')
BINARY_SEARCH_STEPS=$(yq '.configs.binary_search_steps[]' 'configurations/carlini_wagner.yaml')

for model in $MODELS
do
    for size in $SIZES
    do 
        for batch in $BATCHES
        do 
            for n_classes in $N_CLASSES
            do 
                for max in $MAX_ITERATIONS
                do
                    for bs in $BINARY_SEARCH_STEPS
                    do
                        echo "Running settings: \n \
                        > model: $model \n \
                        > size:  $size \n \
                        > batch: $batch \n \
                        > n_classes: $n_classes \n \
                        > binary_search_steps: $bs \n \
                        > max_iterations: $max   "
                        python -m src.attacks.white_box.carlini_wagner_l2 \
                        --model $model \
                        --size  $size \
                        --batch $batch \
                        --n_classes $n_classes \
                        --binary_search_steps $bs \
                        --max_iterations $max                
                    done
                done
            done
        done
    done
done