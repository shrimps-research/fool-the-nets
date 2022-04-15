# Fast Gradient Sign Method Runs
MODELS=$(yq '.configs.models[]' 'configurations/pgd.yaml')
SIZES=$(yq '.configs.sizes[]' 'configurations/pgd.yaml')
BATCHES=$(yq '.configs.batches[]' 'configurations/pgd.yaml')
EPSILON=$(yq '.configs.epsilon[]' 'configurations/pgd.yaml')

for model in $MODELS
do
    for size in $SIZES
    do 
        for batch in $BATCHES
        do 
            for epsilon in $EPSILON
            do 
                echo "Running settings: \n \
                > model: $model \n \
                > size:  $size \n \
                > batch: $batch \n \
                > epsilon: $epsilon \n "
                python -m src.attacks.white_box.fgsm \
                --model $model \
                --size  $size \
                --batch $batch \
                --epsilon $epsilon 
            done
        done
    done
done