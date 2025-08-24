bash scripts/env.sh
CKPT=$1
OUTPUT=$2
# torchrun --nnodes=1 \
#          --node_rank=0 \
#          --master_addr=127.0.0.1 \
#          --nproc_per_node=1 \
#          --master_port=63668 \
torchrun --nproc_per_node $MLP_WORKER_GPU \
         --master_addr $MLP_WORKER_0_HOST \
         --node_rank $MLP_ROLE_INDEX \
         --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
         evaluation/inference_agentthink.py --checkpoint $CKPT --output_name $OUTPUT --out-dir results/agentthink