# bash scripts/env.sh
CKPT=$1
OUTPUT=$2
torchrun --nnodes=1 \
         --node_rank=0 \
         --master_addr=127.0.0.1 \
         --nproc_per_node=1 \
         --master_port=63668 \
         evaluation/inference_agentthink.py --checkpoint $CKPT --output_name $OUTPUT --out-dir results/debug_agentthink