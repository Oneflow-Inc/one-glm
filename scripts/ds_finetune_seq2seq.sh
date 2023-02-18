DATA_ROOT="/data/home/fengwen/cnn_dailmail/cnn-dailymail"
CHECKPOINT_PATH="/home/fengwen/one-glm/runs"
SAVE_PATH="/home/fengwen/one-glm/runs"
DATESTR=$(date +"%m-%d-%H-%M")


source $1    # Model
source $2    # Task

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8 # 代表使用GPU数
HOST_FILE_PATH="./hostfile"
MP_SIZE=20
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
# 注意：不接受--deep_speed参数 因为oneflow的launch模块支持单个或者多个节点的启动，不需要使用deepspeed来辅助

# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
# DISTRIBUTED_ARGS="${OPTIONS_NCCL} deepspeed --hostfile ${HOST_FILE_PATH} --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}"
echo "MODEL_ARGS"$MODEL_ARGS 
 
EXPERIMENT_NAME=${EXPERIMENT_NAME}_${DATESTR}
mkdir logs
run_cmd="python  -m oneflow.distributed.launch --nproc_per_node   ${NUM_GPUS_PER_WORKER}  finetune_glm.py \
       --finetune \
       --checkpoint-activations \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${SAVE_PATH} \
       --num-workers 1 \
       --no-load-lr-scheduler \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       --model-parallel-size ${MP_SIZE} \
       --overwrite \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}.txt"

echo ${run_cmd}
eval ${run_cmd}
set +x


