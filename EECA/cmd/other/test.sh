python eeca.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --alpha 0 --theta 2 --coding Full-Entropy --wm_method noise --runname cifar10_resnet34_noise_C10_K20_Full-Entropy --cuda cuda:1
python eeca.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --alpha 0 --theta 2 --coding Tardos-1 --wm_method noise --runname cifar10_resnet34_noise_C10_K20_Tardos-1 --cuda cuda:1
python eeca.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --alpha 0 --theta 2 --coding Tardos-2 --wm_method noise --runname cifar10_resnet34_noise_C10_K20_Tardos-2 --cuda cuda:1

python eeca.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --alpha 0 --theta 2 --coding Full-Entropy --wm_method exponential_weighting --runname cifar10_resnet34_exponential-weighting_C10_K20_Full-Entropy --cuda cuda:1
python eeca.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --alpha 0 --theta 2 --coding Tardos-1 --wm_method exponential_weighting --runname cifar10_resnet34_exponential-weighting_C10_K20_Tardos-1 --cuda cuda:1
python eeca.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --alpha 0 --theta 2 --coding Tardos-2 --wm_method exponential_weighting --runname cifar10_resnet34_exponential-weighting_C10_K20_Tardos-2 --cuda cuda:1

python eeca.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --alpha 0 --theta 2 --coding Full-Entropy --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Full-Entropy --cuda cuda:1
python eeca.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --alpha 0 --theta 2 --coding Tardos-1 --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Tardos-1 --cuda cuda:1
python eeca.py --dataset cifar10 --num_classes 10 --arch resnet34 --K 20 --alpha 0 --theta 2 --coding Tardos-2 --wm_method ood --runname cifar10_resnet34_ood_C10_K20_Tardos-2 --cuda cuda:1
