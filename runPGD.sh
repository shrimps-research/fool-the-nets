# Projected-Gradient Descent Runs
MODELS=$(yq '.configs.models[]' 'configurations/pgd.yaml')
SIZES=$(yq '.configs.sizes[]' 'configurations/pgd.yaml')
BATCHES=$(yq '.configs.batches[]' 'configurations/pgd.yaml')
EPSILON=$(yq '.configs.epsilon[]' 'configurations/pgd.yaml')
ITERATIONS=$(yq '.configs.iterations[]' 'configurations/pgd.yaml')
STEPS=$(yq '.configs.steps[]' 'configurations/pgd.yaml')

for model in $MODELS
do
    for size in $SIZES
    do 
        for batch in $BATCHES
        do 
            for epsilon in $EPSILON
            do 
                for iter in $ITERATIONS
                do
                    for step in $STEPS
                    do
                        echo "Running settings: \n \
                        > model: $model \n \
                        > size:  $size \n \
                        > batch: $batch \n \
                        > epsilon: $epsilon \n \
                        > step: $step \n \
                        > iterations: $iter   "
                        python -m src.attacks.white_box.pgd \
                        --model $model \
                        --size  $size \
                        --batch $batch \
                        --epsilon $epsilon \
                        --step $step \
                        --iterations $iter               
                    done
                done
            done
        done
    done
done