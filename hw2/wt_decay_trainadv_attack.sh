python3 main.py --model=resnet --dataset=cifar10  --weight_decay=1e-5 
python3 main.py --model=resnet --dataset=cifar10  --weight_decay=1e-4
python3 main.py --model=resnet --dataset=cifar10  --weight_decay=1e-3
python3 main.py --model=resnet --dataset=cifar10  --weight_decay=1e-2
python3 main.py --model=resnet --dataset=cifar10  --weight_decay=1e-1

python3 adv_attack.py --model=resnet --dataset=cifar10 --weight_decay=1e-5
python3 adv_attack.py --model=resnet --dataset=cifar10 --weight_decay=1e-4
python3 adv_attack.py --model=resnet --dataset=cifar10 --weight_decay=1e-3
python3 adv_attack.py --model=resnet --dataset=cifar10 --weight_decay=1e-2
python3 adv_attack.py --model=resnet --dataset=cifar10 --weight_decay=1e-1
