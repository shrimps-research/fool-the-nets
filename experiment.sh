# FGSM
# Smaller epsilon is needed to trick ViT compared to the PerceiverIO model
python -m src.attacks.white_box.fgsm --model vit --epsilon 0.01 --size 64 --batch 16
python -m src.attacks.white_box.fgsm --model perceiver-io --epsilon 0.03 --size 64 --batch 16

# PGD
python -m src.attacks.white_box.pgd --model vit --epsilon 0.01 --step 0.005 --iterations 10 --size 64 --batch 16
python -m src.attacks.white_box.pgd --model perceiver-io --epsilon 0.03 --step 0.005 --iterations 10 --size 64 --batch 16

# Transfer attacking
## It seems to work better for ViT as source network and PerceiverIO as target network


### TODO: Fix the following error: 'No module named src.attacks.black_box.pgd_target_attack'
python -m src.attacks.black_box.fgsm_target_attack --source vit --target perceiver-io --epsilon 0.07 --size 64 --batch 16
python -m src.attacks.black_box.pgd_target_attack --source vit --target perceiver-io --epsilon 0.07 --step 0.02 --iterations 10 --size 64 --batch 16

python -m src.attacks.black_box.fgsm_target_attack --source perceiver-io --target vit --epsilon 0.07 --size 64 --batch 16
python -m src.attacks.black_box.pgd_target_attack --source perceiver-io --target vit --epsilon 0.07 --step 0.02 --iterations 10 --size 64 --batch 16