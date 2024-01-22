python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method weight_average --attack_method fine-tune --coding Full-Entropy --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Full-Entropy --cuda cuda:1
python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method weight_average --attack_method fine-tune --coding Tardos-1 --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Tardos-1 --cuda cuda:1
python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method weight_average --attack_method fine-tune --coding Tardos-2 --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Tardos-2 --cuda cuda:1

python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method majority_vote --attack_method fine-tune --coding Full-Entropy --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Full-Entropy --cuda cuda:1
python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method majority_vote --attack_method fine-tune --coding Tardos-1 --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Tardos-1 --cuda cuda:1
python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method majority_vote --attack_method fine-tune --coding Tardos-2 --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Tardos-2 --cuda cuda:1

python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method weight_average --attack_method pruning --coding Full-Entropy --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Full-Entropy --cuda cuda:1
python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method weight_average --attack_method pruning --coding Tardos-1 --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Tardos-1 --cuda cuda:1
python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method weight_average --attack_method pruning --coding Tardos-2 --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Tardos-2 --cuda cuda:1

python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method majority_vote --attack_method pruning --coding Full-Entropy --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Full-Entropy --cuda cuda:1
python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method majority_vote --attack_method pruning --coding Tardos-1 --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Tardos-1 --cuda cuda:1
python other_collusion_attacks.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --collusion_method majority_vote --attack_method pruning --coding Tardos-2 --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Tardos-2 --cuda cuda:1
