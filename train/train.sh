source /gpfs/public/vl/gjs/.bashrc
conda activate sd

cd /gpfs/public/vl/gjs/Long-CLIP/train
export OMP_NUM_THREADS=32
# envs from volces/pai/arsenal/default
export GPUS_PER_NODE=${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}
export TASK_ID=${MLP_TASK_ID:-$(date "+%Y-%m-%d-%H-%M")}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
accelerate launch \
    --config_file fsdp_config.yaml \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank ${NODE_RANK} \
    --num_machines ${NNODES} \
    --num_processes $(expr ${NNODES} \* ${GPUS_PER_NODE}) \
    train_accelerate.py \
    --batch_size 512 \
    --num_epoch 50\
    --base_model "/gpfs/public/vl/gjs/Long-CLIP/checkpoints/longclip-L.pt" \
    --from_checkpoint 