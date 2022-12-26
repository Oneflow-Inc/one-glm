#! /bin/bash
# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

source $1
DATESTR=$(date +"%m-%d-%H-%M")

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="/workspace/hostfile"

mkdir -p logs
# run_cmd="${OPTIONS_NCCL} deepspeed --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_glm.py ${gpt_options} 2>&1 | tee logs/log-${DATESTR}.txt"

_DEVICE_NUM_PER_NODE=2
_MASTER_ADDR=127.0.0.1
_NUM_NODES=1
_NODE_RANK=0

mkdir logs
run_cmd="python3 -m oneflow.distributed.launch \
    --nproc_per_node $_DEVICE_NUM_PER_NODE \
    --nnodes $_NUM_NODES \
    --node_rank $_NODE_RANK \
    --master_addr $_MASTER_ADDR \
    run_test.py ${gpt_options} \
    2>&1 | tee logs/log-${DATESTR}.txt"


echo ${run_cmd}
eval ${run_cmd}

set +x