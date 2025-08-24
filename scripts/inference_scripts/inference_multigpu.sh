bash scripts/env.sh
CKPT=$1
OUTPUT=$2
# nproc_per_node：单个实例（机器）上运行的进程数，使用 GPU 时通常为每台机器上的 GPU 数量。
# nnodes：对应环境变量 MLP_WORKER_NUM 的值。
# node_rank：对应环境变量 MLP_ROLE_INDEX 的值。
# master_addr：对应环境变量 MLP_WORKER_0_HOST 的值。
# master_port：对应环境变量 MLP_WORKER_0_PORT 的值。
torchrun --nproc_per_node $MLP_WORKER_GPU \
         --master_addr $MLP_WORKER_0_HOST \
         --node_rank $MLP_ROLE_INDEX \
         --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
        evaluation/inference.py --checkpoint $CKPT --output_name $OUTPUT