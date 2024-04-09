GPU_NUM=4
BATCHSIZE=8
LOG_NAME="ours-1-8"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export WANDB_MODE=offline
export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1
torchrun --master_port=29442 --nproc_per_node=$GPU_NUM train_x4k.py --batch_size $BATCHSIZE --need_patch --train_data_path ./datasets/X4K1000FPS/train --val_data_path ./datasets/X4K1000FPS/val
#torchrun --nproc_per_node=$GPU_NUM --master_port=29499 train.py --batch_size $BATCHSIZE --data_path ./vimeo_triplet --wandb_log

# #-x SH-IDC1-10-5-36-43,SH-IDC1-10-5-36-85,SH-IDC1-10-5-36-86,SH-IDC1-10-5-36-143,SH-IDC1-10-5-36-95 \