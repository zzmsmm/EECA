python attack.py --dataset cifar10 --num_classes 10 --arch resnet34 --attack_method fine-tune --K 20 --coding Full-Entropy --wm_method noise --runname cifar10_resnet34_noise_C10_K20_Full-Entropy --cuda cuda:0
#python attack.py --dataset cifar10 --num_classes 10 --arch resnet34 --attack_method pruning --K 20 --coding Full-Entropy --wm_method noise --runname cifar10_resnet34_noise_C10_K20_Full-Entropy --cuda cuda:0

#python attack.py --dataset cifar10 --num_classes 10 --arch resnet34 --attack_method fine-tune --K 20 --coding Full-Entropy --wm_method exponential_weighting --runname cifar10_resnet34_exponential-weighting_C10_K20_Full-Entropy --cuda cuda:0
#python attack.py --dataset cifar10 --num_classes 10 --arch resnet34 --attack_method pruning --K 20 --coding Full-Entropy --wm_method exponential_weighting --runname cifar10_resnet34_exponential-weighting_C10_K20_Full-Entropy --cuda cuda:0

#python attack.py --dataset cifar10 --num_classes 10 --arch resnet34 --attack_method fine-tune --K 20 --coding Full-Entropy --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Full-Entropy --cuda cuda:0
#python attack.py --dataset cifar10 --num_classes 10 --arch resnet34 --attack_method pruning --K 20 --coding Full-Entropy --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Full-Entropy --cuda cuda:0

python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method weight_average --attack_method fine-tune --coding Full-Entropy --wm_method noise --runname cifar10_resnet34_noise_C10_K20_Full-Entropy --cuda cuda:0
python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method majority_vote --attack_method fine-tune --coding Full-Entropy --wm_method noise --runname cifar10_resnet34_noise_C10_K20_Full-Entropy --cuda cuda:0
