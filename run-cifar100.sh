##############
## CIFAR100 ##
##############

# Straggler Learning
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/6_6_6/image_cifar100_vit_fedavg_depthfl-6_6_6_iid.yaml'
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/6_6_6/image_cifar100_vit_fedavg_depthfl-6_6_6_noniid-pat_10_dir.yaml'
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/6_6_6/image_cifar100_vit_fedavg_depthfl-6_6_6_noniid-pat_20_dir.yaml'

# Exclusive Learning
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/0_0_12/image_cifar100_vit_fedavg_depthfl-0_0_12_iid.yaml'
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/0_0_12/image_cifar100_vit_fedavg_depthfl-0_0_12_noniid-pat_10_dir.yaml'
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/0_0_12/image_cifar100_vit_fedavg_depthfl-0_0_12_noniid-pat_20_dir.yaml'

# Fed-HeLLo-Bone
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50.yaml' # Fed-HELLo Bone
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_20_dir-noprior-s50-e50.yaml' # Fed-HELLo Bone
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_10_dir-noprior-s50-e50.yaml' # Fed-HELLo Bone

# Similar for other datasets ....