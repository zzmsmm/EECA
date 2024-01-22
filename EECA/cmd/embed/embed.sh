for ((i=1;i<=5;i++));do
python embed_watermarks.py --method ProtectingIP --embed_type pretrained --wm_type noise --loadmodel cifar10_resnet34_clean --dataset cifar10 --num_classes 10 --arch resnet34 --epochs_w_wm 20 --batch_size 64 --wm_batch_size 16 --lr 0.0001 --optim SGD --sched CosineAnnealingLR --patience 20 --runname cifar10_resnet34_noise_C10_K5_Full-Entropy_$i --K 5 --coding Full-Entropy --trg_set_sizes_list 400 --cuda cuda:2
done
for ((i=1;i<=5;i++));do
python embed_watermarks.py --method ProtectingIP --embed_type pretrained --wm_type noise --loadmodel cifar10_resnet34_clean --dataset cifar10 --num_classes 10 --arch resnet34 --epochs_w_wm 20 --batch_size 64 --wm_batch_size 16 --lr 0.0001 --optim SGD --sched CosineAnnealingLR --patience 20 --runname cifar10_resnet34_noise_C10_K5_Tardos-1_$i --K 5 --coding Tardos-1 --trg_set_sizes_list 400 --cuda cuda:2
done
for ((i=1;i<=5;i++));do
python embed_watermarks.py --method ProtectingIP --embed_type pretrained --wm_type noise --loadmodel cifar10_resnet34_clean --dataset cifar10 --num_classes 10 --arch resnet34 --epochs_w_wm 20 --batch_size 64 --wm_batch_size 16 --lr 0.0001 --optim SGD --sched CosineAnnealingLR --patience 20 --runname cifar10_resnet34_noise_C10_K5_Tardos-2_$i --K 5 --coding Tardos-2 --trg_set_sizes_list 400 --cuda cuda:2
done

for ((i=1;i<=5;i++));do
python embed_watermarks.py --method ExponentialWeighting --embed_type pretrained --loadmodel cifar10_resnet34_clean --dataset cifar10 --num_classes 10 --arch resnet34 --epochs_w_wm 20 --batch_size 64 --wm_batch_size 16 --lr 0.0001 --optim SGD --sched CosineAnnealingLR --patience 20 --runname cifar10_resnet34_exponential-weighting_C10_K5_Full-Entropy_$i --K 5 --coding Full-Entropy --trg_set_sizes_list 400 --cuda cuda:2
done
for ((i=1;i<=5;i++));do
python embed_watermarks.py --method ExponentialWeighting --embed_type pretrained --loadmodel cifar10_resnet34_clean --dataset cifar10 --num_classes 10 --arch resnet34 --epochs_w_wm 20 --batch_size 64 --wm_batch_size 16 --lr 0.0001 --optim SGD --sched CosineAnnealingLR --patience 20 --runname cifar10_resnet34_exponential-weighting_C10_K5_Tardos-1_$i --K 5 --coding Tardos-1 --trg_set_sizes_list 400 --cuda cuda:2
done
for ((i=1;i<=5;i++));do
python embed_watermarks.py --method ExponentialWeighting --embed_type pretrained --loadmodel cifar10_resnet34_clean --dataset cifar10 --num_classes 10 --arch resnet34 --epochs_w_wm 20 --batch_size 64 --wm_batch_size 16 --lr 0.0001 --optim SGD --sched CosineAnnealingLR --patience 20 --runname cifar10_resnet34_exponential-weighting_C10_K5_Tardos-2_$i --K 5 --coding Tardos-2 --trg_set_sizes_list 400 --cuda cuda:2
done

for ((i=1;i<=5;i++));do
python embed_watermarks.py --method Ood --embed_type pretrained --loadmodel cifar10_resnet34_clean --dataset cifar10 --num_classes 10 --arch resnet34 --epochs_w_wm 20 --batch_size 64 --wm_batch_size 16 --lr 0.0001 --optim SGD --sched CosineAnnealingLR --patience 20 --runname cifar10_resnet34_ood_C10_K5_Full-Entropy_$i --K 5 --coding Full-Entropy --trg_set_sizes_list 400 --cuda cuda:2
done
for ((i=1;i<=5;i++));do
python embed_watermarks.py --method Ood --embed_type pretrained --loadmodel cifar10_resnet34_clean --dataset cifar10 --num_classes 10 --arch resnet34 --epochs_w_wm 20 --batch_size 64 --wm_batch_size 16 --lr 0.0001 --optim SGD --sched CosineAnnealingLR --patience 20 --runname cifar10_resnet34_ood_C10_K5_Tardos-1_$i --K 5 --coding Tardos-1 --trg_set_sizes_list 400 --cuda cuda:2
done
for ((i=1;i<=5;i++));do
python embed_watermarks.py --method Ood --embed_type pretrained --loadmodel cifar10_resnet34_clean --dataset cifar10 --num_classes 10 --arch resnet34 --epochs_w_wm 20 --batch_size 64 --wm_batch_size 16 --lr 0.0001 --optim SGD --sched CosineAnnealingLR --patience 20 --runname cifar10_resnet34_ood_C10_K5_Tardos-2_$i --K 5 --coding Tardos-2 --trg_set_sizes_list 400 --cuda cuda:2
done